"""
A federated learning server using FedBuff.

Reference:

Nguyen, J., Malik, K., Zhan, H., et al., "Federated Learning with Buffered Asynchronous Aggregation,
" in Proc. International Conference on Artificial Intelligence and Statistics (AISTATS 2022).

https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf
"""
import asyncio

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        total_updates = len(updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for update in deltas_received:
            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (1 / total_updates)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update
