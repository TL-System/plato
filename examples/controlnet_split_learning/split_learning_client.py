"""
A federated learning server using split learning.

Reference:

Vepakomma, et al., "Split Learning for Health: Distributed Deep Learning without Sharing
Raw Patient Data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf

Chopra, Ayush, et al. "AdaSplit: Adaptive Trade-offs for Resource-constrained Distributed
Deep Learning." arXiv preprint arXiv:2112.01637 (2021).

https://arxiv.org/pdf/2112.01637.pdf
"""
# pylint:disable=import-error
import logging
from types import SimpleNamespace
import time
import torch


from split_learning import split_learning_client
from plato.config import Config


# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-few-public-methods
# pylint:disable=too-many-arguments
class Client(split_learning_client.Client):
    """The split learning client."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.iter_left = self.iterations

    async def inbound_processed(self, processed_inbound_payload):
        """Extract features or complete the training using split learning."""
        server_payload, info = processed_inbound_payload

        # Preparing the client response
        report, payload = None, None

        if info == "prompt":
            # Server prompts a new client to conduct split learning
            self._load_context(self.client_id)
            report, payload = self._extract_features()
            if "cuda" in Config().device():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.reset_max_memory_allocated()
                report.gpu_mem = gpu_mem
        elif info == "gradients":
            # server sends the gradients of the features, i.e., complete training
            logging.warning("[%s] Gradients received, complete training.", self)
            training_time, weights = self._complete_training(server_payload)
            self.iter_left -= 1

            if self.iter_left == 0:
                logging.warning(
                    "[%s] Finished training, sending weights to the server.", self
                )
                # Send weights to server for evaluation
                report = SimpleNamespace(
                    client_id=self.client_id,
                    num_samples=self.sampler.num_samples(),
                    accuracy=0,
                    training_time=training_time,
                    comm_time=time.time(),
                    update_response=False,
                    type="weights",
                )
                payload = weights
                self.iter_left = self.iterations
            else:
                # Continue feature extraction
                report, payload = self._extract_features()
                report.training_time += training_time
                if "cuda" in Config().device():
                    gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    torch.cuda.reset_max_memory_allocated()
                    report.gpu_mem = gpu_mem

            # Save the state of current client
            self._save_context(self.client_id)
        return report, payload
