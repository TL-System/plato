"""
A federated learning server using federated averaging to train GAN models.
"""
import asyncio

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.extract_client_updates(updates)

        # Total sample is the same for both Generator and Discriminator
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates])

        # Perform weighted averaging for both Generator and Discriminator
        gen_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][0].items()
        }
        disc_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][1].items()
        }

        for i, update in enumerate(weights_received):
            __, report, __, __ = updates[i]
            num_samples = report.num_samples

            update_from_gen, update_from_disc = update

            for name, delta in update_from_gen.items():
                gen_avg_update[name] += delta * (num_samples /
                                                 self.total_samples)

            for name, delta in update_from_disc.items():
                disc_avg_update[name] += delta * (num_samples /
                                                  self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return gen_avg_update, disc_avg_update

    def customize_server_payload(self, payload):
        """ Customize the server payload before sending to the client. """
        weights_gen = None
        weights_disc = None
        if hasattr(Config().server, 'network_to_sync'):
            if hasattr(Config().server.network_to_sync,
                       'generator') and Config().server.network_to_sync.generator:
                weights_gen = payload[0]
            if hasattr(Config().server.network_to_sync,
                       'discriminator') and Config().server.network_to_sync.discriminator:
                weights_disc = payload[1]
        return weights_gen, weights_disc
