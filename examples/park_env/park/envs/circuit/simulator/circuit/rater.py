import numpy as np

from park.envs.circuit.simulator.utility.misc import AttrDict

__all__ = ['Rater']


class Rater(object):
    __SCALES__ = {
        'linear': lambda a, b: a / b,
        'log': lambda a, b: np.log10(a) / np.log10(b)
    }

    def __init__(self, score_fail, *, centralized_target=False):
        self._score_fail = score_fail
        self._score_unsaturated = None
        self._metrics = dict()
        self._centralized = centralized_target

    def set_unsaturated_score(self, score):
        self._score_unsaturated = score
        return self

    def metric(self, key, *, scale, direction, constrained, targeted):
        assert scale in ('linear', 'log')
        assert direction in ('minimize', 'maximize')
        assert isinstance(targeted, bool)
        assert constrained or targeted
        self._metrics[key] = AttrDict(
            scale=scale,
            direction=direction,
            constrained=constrained,
            targeted=targeted
        )
        return self

    @property
    def constrained(self):
        return {key: option for key, option in self._metrics.items()
                if not isinstance(option.constrained, bool) or option.constrained}

    @property
    def targeted(self):
        return {key: option for key, option in self._metrics.items() if option.targeted}

    def __str__(self):
        result = 'Benchmark(\n'
        result += f'    Constrained: {sorted([key for key in self.constrained])}\n'
        result += f'    Targeted: {sorted([key for key in self.targeted])}\n'
        result += f'    Centralized: {self._centralized}\n'
        result += f'    Metrics:\n'
        for key, option in self._metrics.items():
            result += f'        {option.direction} "{key}" in {option.scale} scale\n'
        result += ')'
        return result

    def __repr__(self):
        return self.__str__()

    def __call__(self, values, result, specs):
        if result is None:
            return self._score_fail
        metrics = result.metrics
        metrics = AttrDict(**metrics)
        specs = AttrDict(**specs)
        if self._score_unsaturated is not None and not metrics.saturated:
            return self._score_unsaturated
        satisfied = True
        for key, option in self.constrained.items():
            metric = getattr(metrics, key)
            spec = getattr(specs, key) if isinstance(option.constrained, bool) else option.constrained
            if option.direction == 'minimize' and metric > spec:
                satisfied = False
                break
            if option.direction == 'maximize' and metric < spec:
                satisfied = False
                break

        score = 0
        if not satisfied:
            for key, option in self.constrained.items():
                metric = getattr(metrics, key)
                spec = getattr(specs, key) if isinstance(option.constrained, bool) else option.constrained
                scale = self.__SCALES__[option.scale]
                if option.direction == 'minimize':
                    score += scale(spec, max(metric, spec)) - 1
                else:
                    score += scale(min(metric, spec), spec) - 1
        else:
            for key, option in self.targeted.items():
                metric = getattr(metrics, key)
                spec = getattr(specs, key)
                scale = self.__SCALES__[option.scale]
                if option.direction == 'minimize':
                    score += scale(spec, metric)
                else:
                    score += scale(metric, spec)
                if isinstance(self._centralized, bool) and self._centralized:
                    score -= 1
            if not isinstance(self._centralized, bool):
                score += self._centralized

        return score
