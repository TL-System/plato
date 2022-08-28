import numpy as np
import pickle
import sklearn.linear_model as lm

def get_single(val):
    try:
        return get_single(val[0])
    except Exception as e:
        return val

class LinearModel:
    def __init__(self, xs=None, ys=None):
        self.coef_ = 0
        self.intcpt_ = 0
        if xs is not None and ys is not None:
            if len(xs) == 2:
                self.fit_exact(xs, ys)
            else:
                self.best_fit(xs, ys)

    def fit_exact(self, xs, ys):
        assert xs is not None and len(xs) == 2
        assert ys is not None and len(ys) == 2
        if np.abs(xs[1] - xs[0]) < 1e-10:
            self.coef_ = 0
            self.intcpt_ = (ys[0] + ys[1])/2
            return
        self.coef_ = (ys[1] - ys[0]) / (xs[1] - xs[0])
        self.intcpt_ = ys[0] - self.coef_ * xs[0]

    def best_fit(self, xs, ys):
        assert len(xs) == len(ys)
        m = lm.LinearRegression()
        m.fit(np.array(xs).reshape(-1, 1), np.array(ys).reshape(-1, 1))
        self.coef_ = get_single(m.coef_[0])
        self.intcpt_ = get_single(m.intercept_)

    # x can be a scalar or numpy array
    def predict(self, x):
        return self.coef_ * x + self.intcpt_

    def inverse(self, y):
        if self.coef_ == 0:
            return self.intcpt_
        return (y - self.intcpt_) / self.coef_

    # xs and ys should be numpy arrays of the same dimension
    # Returns the sum of squared errors for all the given points
    def sse(self, xs, ys):
        got = self.predict(xs)
        return np.sum(np.square(got - ys))

    # xs and ys should be numpy arrays of the same dimension
    # Returns the sum of absolute errors for all the given points
    def sae(self, xs, ys):
        got = self.predict(xs).reshape(-1)
        return np.sum(np.abs(got - ys))


class MonotonicRMI:
    def __init__(self, experts, last_layer_monotonic=False):
        self.models = []
        self.experts = experts
        self.last_layer_monotonic = last_layer_monotonic
        if experts[0] != 1:
            self.experts = [1] + self.experts
        self.minval = 0
        self.maxval = 0
        # The first integer value in the domain of each segment.
        self.domain_separators = []

    def fit(self, xs, ys, verbose=True):
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        self.minval = ys.min()
        self.maxval = ys.max()
        # To fit, we split the space evenly into n parts.
        # For each part, fit a line from the starting to the ending point and compute the total
        # error of the fit for all points it contains (its bucket).
        # Then, allocate models to each bucket proportional to that bucket's error.
        splits = np.linspace(0, len(xs), self.experts[0] + 1).astype(int)
        splits = np.array(list(splits)[:-1] + [len(xs)-1], dtype=int)
        for level, n in enumerate(self.experts):
            if verbose:
                print(f'Fitting monotonic RMI layer {level} with {n} experts')
            saes = [0]
            for i in range(len(splits)-1):
                # This isn't the final model. It's just a measure of how well our current split
                # approximates the true CDF.
                #model = LinearModel(xs=[xs[splits[i]], xs[splits[i+1]]],
                #        ys=[ys[splits[i]], ys[splits[i+1]]])
                #err = model.sse(xs[splits[i]:splits[i+1]], ys[splits[i]:splits[i+1]])
                saes.append(splits[i+1] - splits[i])
            if level == len(self.experts) - 1:
                # Fit the last layer
                model_level = []
                for i in range(len(splits)-1):
                    model = None
                    self.domain_separators.append(xs[splits[i]])
                    if self.last_layer_monotonic:
                        model = LinearModel(xs=[xs[splits[i]], xs[splits[i+1]]],
                                ys=[ys[splits[i]], ys[splits[i+1]]])
                    else:
                        xvals = xs[splits[i]:splits[i+1]]
                        yvals = ys[splits[i]:splits[i+1]]
                        if splits[i] >= splits[i+1]:
                            xvals = [xs[splits[i]], xs[splits[i]]]
                            yvals = [ys[splits[i]], ys[splits[i]]]
                        model = LinearModel(xs=xvals, ys=yvals)
                    model_level.append(model)
                self.domain_separators.append(xs.max()+1)
                self.models.append(model_level)
                return

            target_range = 1
            ranges = ys[splits]
            if level < len(self.experts)-1:
                target_range = self.experts[level+1]
                ranges = (target_range - 0.5*(1.0/len(ys))) * np.cumsum(saes) / sum(saes)
            prev_range = 0
            model_level = []
            next_split_ix = 1
            new_splits_x = []
            for i, r in enumerate(ranges[1:]):
                model = LinearModel(xs=[xs[splits[i]], xs[splits[i+1]]], ys=[ranges[i], r])
                #print('Fitting model with:', splits[i], splits[i+1], xs[splits[i]], xs[splits[i+1]], ranges[i], r)
                prev_range = r
                model_level.append(model)
                while next_split_ix < r:
                    new_splits_x.append(model.inverse(next_split_ix))
                    #print(next_split_ix, model.inverse(next_split_ix), 'model:', model.coef_,
                    #        model.intcpt_)
                    next_split_ix += 1
            splits = np.searchsorted(xs, new_splits_x).astype(int)
            splits = np.array([0] + list(splits) + [len(xs)-1])
            self.models.append(model_level)

    def predict(self, xs):
        xs = xs.reshape(-1)
        buckets = [np.arange(0, len(xs)).astype(int)]
        for level, model_level in enumerate(self.models):
            # print(len(model_level), len(buckets))
            assert len(model_level) == len(buckets)
            if level < len(self.models) - 1:
                new_buckets = [[] for _ in range(len(self.models[level+1]))]
                for i, bucket in enumerate(buckets):
                    evals = np.clip(model_level[i].predict(xs[bucket]).astype(int), 0, len(self.models[level+1])-1)
                    for nb in range(len(new_buckets)):
                        match_ixs = np.where(evals == nb)[0]
                        new_buckets[nb].extend(bucket[match_ixs])
                buckets = []
                for nb in new_buckets:
                    buckets.append(np.array(nb).astype(int))
            else:
                results = np.zeros(len(xs))
                for i, bucket in enumerate(buckets):
                    results[bucket] = model_level[i].predict(xs[bucket])
                return results

    def predict_single(self, x):
        y = 0
        for layer in self.models:
            mapped = max(0, min(len(layer) - 1, int(y)))
            y = layer[mapped].predict(x)
        return min(self.maxval, max(self.minval, y))

    def dump(self, filename):
        # print('Test probe: layer 2, model 46: (%f, %f)' % \
        #         (self.models[2][46].coef_, self.models[2][46].intcpt_))
        sizes = np.array([len(x) for x in self.models], dtype='int32')
        weights = []
        for layer in self.models:
            for model in layer:
                weights.append(model.coef_)
                weights.append(model.intcpt_)
        weights = np.array(weights, dtype='float64')
        
        with open(filename, 'wb+') as f:
            f.write(bytes('%d\n' % len(self.experts), 'utf-8'))
            f.write(sizes.tobytes())
            f.write(weights.tobytes())
            f.write(np.array(self.domain_separators, dtype=np.int64).tobytes())

        pickle.dump(self, open(filename.split('.')[0] + '.pkl', 'wb+'))

