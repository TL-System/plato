"""The YOLOV8 model for PyTorch."""
import logging

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """The YOLOV8 trainer."""

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The training loop for YOLOv8.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        """
        self.model.train(
            data=Config().data.data_params,
            epochs=Config().trainer.epochs,
        )

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def test_model(self, config, testset, sampler=None, **kwargs):
        """The test loop for YOLOv8.

        Arguments:
        config: A dictionary of configuration parameters.
        testset: The test dataset.
        """

        logging.info("[%s] Started model testing.", self)
        metrics = self.model.val(
            data=Config().data.data_params,
        )

        return metrics.box.map50
