"""
A federated learning server using federated averaging to train GAN models.
"""
import asyncio

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """ Federated learning server using federated averaging to train GAN models. """

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.compute_weight_deltas(updates)

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
        """
        Customize the server payload before sending to the client.

        At the end of each round, the server can choose to only send the global Generator
        or Discriminator (or both or neither) model to the clients next round.

        Reference this paper for more detail:
        https://deepai.org/publication/federated-generative-adversarial-learning

        By default, both model will be sent to the clients.
        """
        weights_gen, weights_disc = payload
        if hasattr(Config().server, 'network_to_sync'):
            if hasattr(Config().server.network_to_sync, 'generator'
                       ) and not Config().server.network_to_sync.generator:
                weights_gen = None
            if hasattr(Config().server.network_to_sync, 'discriminator'
                       ) and not Config().server.network_to_sync.discriminator:
                weights_disc = None
        return weights_gen, weights_disc
