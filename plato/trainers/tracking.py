"""
Keeping a history of metrics during the training run.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, DefaultDict


class RunHistory:
    """
    A simple history of metrics during a training or evaluation run.
    """

    def __init__(self) -> None:
        self._metrics: DefaultDict[str, list[Any]] = defaultdict(list)

    def get_metric_names(self) -> Iterable[str]:
        """
        Returns an iterable set containing of all unique metric names which are
        being tracked.

        :return: an iterable of the unique metric names.
        """
        return set(self._metrics.keys())

    def get_metric_values(self, metric_name: str) -> Sequence[Any]:
        """
        Returns an ordered iterable list of values that has been stored since
        the last reset corresponding to the provided metric name.

        :param metric_name: the name of the metric being tracked.
        :return: an ordered iterable of values that have been recorded for that metric.
        """
        return self._metrics[metric_name]

    def get_latest_metric(self, metric_name: str) -> Any:
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

    def update_metric(self, metric_name: str, metric_value: Any) -> None:
        """
        Records a new value for the given metric.

        :param metric_name: the name of the metric being tracked.
        :param metric_value: the value to record.
        """
        self._metrics[metric_name].append(metric_value)

    def reset(self) -> None:
        """
        Resets the state of the :class:`RunHistory`.

        """
        self._metrics = defaultdict(list)  # type: DefaultDict[str, list[Any]]


class LossTracker:
    """A simple tracker for computing the average loss."""

    def __init__(self) -> None:
        self.loss_value: Any = 0
        self._average: Any = 0
        self.total_loss: Any = 0
        self.running_count = 0

    def reset(self) -> None:
        """Resets this loss tracker."""

        self.loss_value = 0
        self._average = 0
        self.total_loss = 0
        self.running_count = 0

    def update(self, loss_batch_value: Any, batch_size: int = 1) -> None:
        """Updates the loss tracker with another loss value from a batch."""

        self.loss_value = loss_batch_value
        self.total_loss += loss_batch_value * batch_size
        self.running_count += batch_size
        self._average = self.total_loss / self.running_count

    @property
    def average(self) -> float:
        """Returns the computed average of loss values tracked."""
        if isinstance(self._average, (int, float)):
            return float(self._average)
        return float(self._average.cpu().detach().mean().item())
