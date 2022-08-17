"""
A federated learning server using federated averaging to train GAN models.
"""
import asyncio

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """Federated learning server using federated averaging to train GAN models."""

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Total sample is the same for both Generator and Discriminator
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging for both Generator and Discriminator
        gen_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0][0].items()
        }
        disc_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0][1].items()
        }

        for i, update in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples

            update_from_gen, update_from_disc = update

            for name, delta in update_from_gen.items():
                gen_avg_update[name] += delta * (num_samples / self.total_samples)

            for name, delta in update_from_disc.items():
                disc_avg_update[name] += delta * (num_samples / self.total_samples)

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
        if hasattr(Config().server, "network_to_sync"):
            network = Config().server.network_to_sync.lower()
        else:
            network = "both"

        weights_gen, weights_disc = payload
        if network == "none":
            server_payload = None, None
        elif network == "generator":
            server_payload = weights_gen, None
        elif network == "discriminator":
            server_payload = None, weights_disc
        elif network == "both":
            server_payload = payload
        else:
            raise ValueError(f"Unknown value to attribute network_to_sync: {network}")

        return server_payload
