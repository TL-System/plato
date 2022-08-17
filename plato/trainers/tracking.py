"""
Keeping a history of metrics during the training run.
"""
from collections import defaultdict
from typing import Iterable


class RunHistory:
    """
    A simple history of metrics during a training or evaluation run.
    """

    def __init__(self):
        self._metrics = defaultdict(list)

    def get_metric_names(self) -> Iterable:
        """
        Returns an iterable set containing of all unique metric names which are
        being tracked.

        :return: an iterable of the unique metric names.
        """
        return set(self._metrics.keys())

    def get_metric_values(self, metric_name) -> Iterable:
        """
        Returns an ordered iterable list of values that has been stored since
        the last reset corresponding to the provided metric name.

        :param metric_name: the name of the metric being tracked.
        :return: an ordered iterable of values that have been recorded for that metric.
        """
        return self._metrics[metric_name]

    def get_latest_metric(self, metric_name):
        """
        Returns the most recent value that has been recorded for the given metric.

        :param metric_name: the name of the metric being tracked.
        :return: the last recorded value.
        """
        if len(self._metrics[metric_name]) > 0:
            return self._metrics[metric_name][-1]
        else:
            raise ValueError(
                f"No values have been recorded for the metric {metric_name}"
            )

    def update_metric(self, metric_name, metric_value):
        """
        Records a new value for the given metric.

        :param metric_name: the name of the metric being tracked.
        :param metric_value: the value to record.
        """
        self._metrics[metric_name].append(metric_value)

    def reset(self):
        """
        Resets the state of the :class:`RunHistory`.

        """
        self._metrics = defaultdict(list)


class LossTracker:
    """A simple tracker for computing the average loss."""

    def __init__(self):
        self.loss_value = 0
        self._average = 0
        self.total_loss = 0
        self.running_count = 0

    def reset(self):
        """Resets this loss tracker."""

        self.loss_value = 0
        self._average = 0
        self.total_loss = 0
        self.running_count = 0

    def update(self, loss_batch_value, batch_size=1):
        """Updates the loss tracker with another loss value from a batch."""

        self.loss_value = loss_batch_value
        self.total_loss += loss_batch_value * batch_size
        self.running_count += batch_size
        self._average = self.total_loss / self.running_count

    @property
    def average(self):
        """Returns the computed average of loss values tracked."""

        return self._average.cpu().detach().mean().item()
