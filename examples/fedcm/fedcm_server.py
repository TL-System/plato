import logging
from collections import OrderedDict

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm."""

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        deltas = await self.aggregate_deltas(self.updates)
        self.trainer.delta = deltas.copy()
        updated_weights = self.algorithm.update_weights(
            deltas,
            Config().algorithm.global_learning_rate
            * (Config().algorithm.lr_decay ** self.current_round),
        )
        self.algorithm.load_weights(updated_weights)

        # lr = Config().parameters.optimizer.lr * (Config().algorithm.lr_decay ** self.current_round)
        # for name, weight in self.trainer.delta.items():
        #     self.trainer.delta[name] = self.trainer.delta[name] / lr

        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            # Testing the updated model directly at the server
            self.accuracy = await self.trainer.test(self.testset, self.testset_sampler)

        if hasattr(Config().trainer, "target_perplexity"):
            logging.info("[%s] Global model perplexity: %.2f\n", self, self.accuracy)
        else:
            logging.info(
                "[%s] Global model accuracy: %.2f%%\n", self, 100 * self.accuracy
            )

        self.clients_processed()

    def customize_server_payload(self, payload):
        return [payload, self.trainer.delta, self.current_round]
