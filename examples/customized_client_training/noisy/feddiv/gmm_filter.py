import os
import numpy as np
import sklearn.mixture
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg

from sklearn import datasets, preprocessing
import warnings
import logging
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

class GlobalFilterManager():
    def __init__(self, components=2, local_epochs=3, seed=False, init_params='random'):
        self.random_state = None
        if seed:
            self.random_state = int(seed)
        
        # Initialize GaussianMixture model
        self.model = GaussianMixture(
            X=None,
            n_components=components,
            random_state=self.random_state,
            is_quiet=True,
            init_params=init_params,
        )

        self.init_params = init_params
        self.components = components
        self.local_epochs = local_epochs
        self.selected_clients = {}

    def _set_parameters_from_clients_models(self, server_cached_local_filter):
        # Extract parameters from client models
        self.clients_means = []
        self.clients_covariances = []
        self.clients_weights = []

        for filter_update in server_cached_local_filter:
            parameters = filter_update['parameters']
            self.clients_means.append(parameters['means'][-1])
            self.clients_covariances.append(parameters['covariances'][-1])
            self.clients_weights.append(parameters['weights'][-1])

        # Convert to NumPy arrays
        self.clients_means = np.array(self.clients_means)
        self.clients_covariances = np.array(self.clients_covariances)
        self.clients_weights = np.array(self.clients_weights)

        return

    def set_parameters_from_clients_models(self, server_cached_local_filter):
        # Same as _set_parameters_from_clients_models function
        self._set_parameters_from_clients_models(server_cached_local_filter)

        return
    
    def start_round(self, selected_clients, server_cached_local_filter):
        # Fit the model for selected clients
        for client in selected_clients:
            server_cached_local_filter[client.id] = client.fit(self.model, self.local_epochs)
        return server_cached_local_filter

    # def weighted_average_clients_models(self, dict_len):
    #     dict_len = np.array(dict_len)
    #     gamma = dict_len * 1.0 / np.sum(dict_len)

    #     # Calculate weighted average model parameters
    #     for k in range(len(dict_len)):
    #         self.clients_means[k] = self.clients_means[k] * pow(gamma[k], 1)
    #         self.clients_covariances[k] = self.clients_covariances[k] * pow(gamma[k], 2)
    #         self.clients_covariances * pow(gamma, 2)
    #         self.clients_weights[k] = self.clients_weights[k] * pow(gamma[k], 1)

    #     # Update model parameters
    #     self._update_model_parameters()

    #     return

    def update_server_model(self):
        # Update server-side model
        self.model = GaussianMixture(
            X=None,
            n_components=self.components,
            random_state=self.random_state,
            is_quiet=True,
            init_params=self.init_params,
            weights_init=self.avg_clients_weights,
            means_init=self.avg_clients_means,
            precisions_init=self.avg_clients_precisions
        )
        logging.info(f"GMM stats: mean = {self.avg_clients_means}, weights = {self.avg_clients_weights}, precision = {self.avg_clients_precisions}.")

    def _update_model_parameters(self):
        # Update model parameters
        self.avg_clients_precisions_cholesky = self.model.compute_precision_cholesky(
            self.avg_clients_covariances, self.model.covariance_type
        )
        
        params = (self.avg_clients_weights, self.avg_clients_means, self.avg_clients_covariances, self.avg_clients_precisions_cholesky)
        self.model.set_parameters(params)

        self.avg_clients_precisions = self.model.precisions_

        return
    

    def weighted_average_clients_models(self, dict_len):
        
        dict_len = np.array(dict_len)
        gamma = dict_len * 1.0 / np.sum(dict_len)
        
        for k in range(len(dict_len)):
            self.clients_means[k] = self.clients_means[k] * pow(gamma[k], 1)
            self.clients_covariances[k] = self.clients_covariances[k] * pow(gamma[k], 2)
            self.clients_covariances * pow(gamma, 2)
            self.clients_weights[k] = self.clients_weights[k] * pow(gamma[k], 1)
            
        
        self.avg_clients_means = np.sum(self.clients_means, axis=0)
        self.avg_clients_covariances = np.sum(self.clients_covariances, axis=0)
        self.avg_clients_weights = np.sum(self.clients_weights, axis=0)
        
        self.avg_clients_precisions_cholesky = self.model.compute_precision_cholesky(self.avg_clients_covariances, self.model.covariance_type)
        
        params = (self.avg_clients_weights, self.avg_clients_means, self.avg_clients_covariances, self.avg_clients_precisions_cholesky)
        self.model.set_parameters(params)

        self.avg_clients_precisions = self.model.precisions_

        return


class GaussianMixture(sklearn.mixture.GaussianMixture):

    def __init__(self, X = None, n_components=3, covariance_type='full',
                 weights_init=None, means_init=None, precisions_init=None, covariances_init=None,
                 init_params='kmeans', tol=1e-3, random_state=None, is_quiet=False):

        if not X:
            X = self.gen_filter_data(random_state = 1)

        do_init = (weights_init is None) and (means_init is None) and (precisions_init is None) and (covariances_init is None)
        if do_init:
            _init_model = sklearn.mixture.GaussianMixture(
                n_components=n_components,
                tol=tol,
                covariance_type=covariance_type,
                random_state=random_state,
                init_params=init_params
            )

            # Responsibilities are found through KMeans or randomly assigned, from responsibilities the gaussian parameters are estimated (precisions_ is not calculated)
            _init_model._initialize_parameters(X, np.random.RandomState(random_state))
            # The gaussian parameters are fed into _set_parameters() which computes also precisions_ (the others remain the same)
            _init_model._set_parameters(_init_model._get_parameters())

            weights_init = _init_model.weights_
            means_init = _init_model.means_
            precisions_init = _init_model.precisions_
            covariances_init = _init_model.covariances_

        super().__init__(
            n_components=n_components,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            tol=tol,
            init_params=init_params,
            covariance_type=covariance_type,
            random_state=random_state,
            warm_start=True,
            max_iter=1
        )
        self._is_quiet = is_quiet
        # The gaussian parameters are recomputed by KMeans or randomly, but since the init parameters are given they are discarded (covariances_ is not generated)
        self._initialize_parameters(X, np.random.RandomState(random_state))
        # covariances_ is copied from the initial model (since it has it)
        self.covariances_ = covariances_init
        # precisions_ is computed as before
        self._set_parameters(self._get_parameters())

    def gen_filter_data(self, random_state = None):
        """Generate the initial data for the filter (copied from FedDiv codebase)"""
        if not random_state:
            random_state = 1

        data, labels = datasets.make_blobs(
            n_samples=1000, n_features=1, centers=2, random_state=random_state, shuffle=False
        )

        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        train_dataset = np.array(data)

        lb = preprocessing.LabelBinarizer()
        lb.fit(labels)
        labels = lb.transform(labels)
        init_data = (train_dataset - min(train_dataset)) / (
            max(train_dataset) - min(train_dataset)
        )

        return init_data

    def fit(self, X, epochs=1, labels=None, args=None, output_dir=None):
        self.history_ = {
            'epochs': epochs,
            'converged': [],
            'metrics': {
                'aic': [],
                'bic': [],
                'll': [],
            },
            'parameters': {
                'means': [],
                'covariances': [],
                'weights': []
            }
        }

        if not self._is_quiet:
            self.plot(X, labels, args, output_dir)

        pbar = tqdm(range(epochs), disable=self._is_quiet)
        for epoch in pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch+1, epochs))

            super().fit(X)

            self.history_['converged'].append(self.converged_)

            self.history_['metrics']['aic'].append(self.aic(X))
            self.history_['metrics']['bic'].append(self.bic(X))
            self.history_['metrics']['ll'].append(self.score(X))

            self.history_['parameters']['means'].append(self.means_)
            self.history_['parameters']['weights'].append(self.weights_)
            self.history_['parameters']['covariances'].append(self.covariances_)

            if not self._is_quiet and (epoch+1) % args.plots_step == 0:
                self.plot(X, labels, args, output_dir, 'epoch', epoch)

        if not self._is_quiet:
            if self.converged_:
                print('\nThe model successfully converged.')
            else:
                print('\nThe model did NOT converge.')

        return self.history_

    def predict(self, X):
        predicted_labels = self.predict_proba(X).tolist()
        predicted_labels = np.array(predicted_labels)

        return predicted_labels

    def get_parameters(self):
        parameters = self._get_parameters()

        return parameters

    def set_parameters(self, params):
        self._set_parameters(params)

        return

    def compute_precision_cholesky(self, covariances, covariance_type):
        """Compute the Cholesky decomposition of the precisions.
        Parameters
        ----------
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.
        Returns
        -------
        precisions_cholesky : array-like
            The cholesky decomposition of sample precisions of the current
            components. The shape depends of the covariance_type.
        """
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar.")

        if covariance_type == 'full':
            n_components, n_features, _ = covariances.shape
            precisions_chol = np.empty((n_components, n_features, n_features))
            for k, covariance in enumerate(covariances):
                try:
                    cov_chol = linalg.cholesky(covariance, lower=True)
                except linalg.LinAlgError:
                    raise ValueError(estimate_precision_error_message)
                precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                             np.eye(
                                                                 n_features),
                                                             lower=True).T
        elif covariance_type == 'tied':
            _, n_features = covariances.shape
            try:
                cov_chol = linalg.cholesky(covariances, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                      lower=True).T
        else:
            if np.any(np.less_equal(covariances, 0.0)):
                raise ValueError(estimate_precision_error_message)
            precisions_chol = 1. / np.sqrt(covariances)
        return precisions_chol
