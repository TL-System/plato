"""
HeteroFL algorithm trainer.
"""

import logging
from plato.config import Config
from plato.trainers.basic import Trainer


class ServerTrainer(Trainer):
    """A federated learning trainer of HeteroFL, used by the server."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.is_train = False
        self.model = model(**Config().parameters.model._asdict())

    def test(self, testset, sampler=None, **kwargs) -> float:
        """Because the global model will need to compute the statistics of the model."""
        self.is_train = True
        super().test(testset, sampler, **kwargs)
        self.is_train = False
        return super().test(testset, sampler, **kwargs)

    # pylint: disable=inconsistent-return-statements
    def test_process(self, config, testset, sampler=None, **kwargs):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.
        """
        self.model.to(self.device)
        if self.is_train:
            self.model.train()
        else:
            self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            if sampler is None:
                accuracy = self.test_model(config, testset, **kwargs)
            else:
                accuracy = self.test_model(config, testset, sampler.get(), **kwargs)

        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        if "max_concurrency" in config:
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy
