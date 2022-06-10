"""
A federated learning server using federated averaging to train Actor-Critic models.

To run this example:

python examples/customized/custom_server.py -c examples/customized/server.yml
"""

import logging
from torch import nn
import asyncio
from plato.servers import fedavg
from plato.config import Config


class TD3Server(fedavg.Server):
    """ Federated learning server using federated averaging to train Actor-Critic models. """
    """ A custom federated learning server. """

    def __init__(self, model = None, trainer = None, algorithm = None):
        super().__init__(trainer = trainer, algorithm = algorithm, model = model)
        logging.info("A custom server has been initialized.")
        
    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""

        #print("line 27 in td3_server is being executed")

        weights_received = self.compute_weight_deltas(updates)

        # Total sample is the same for both Generator and Discriminator
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates])

        # Perform weighted averaging for both Generator and Discriminator
        actor_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][0].items()
        }
        critic_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][1].items()
        }
        actor_target_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][2].items()
        }
        critic_target_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][3].items()
        }

        for i, update in enumerate(weights_received):
            __, report, __, __ = updates[i]
            num_samples = report.num_samples

            update_from_actor, update_from_critic, update_from_actor_target, update_from_critic_target = update

            for name, delta in update_from_actor.items():
                actor_avg_update[name] += delta * (num_samples /
                                                 self.total_samples)

            for name, delta in update_from_critic.items():
                critic_avg_update[name] += delta * (num_samples /
                                                  self.total_samples)

            for name, delta in update_from_actor_target.items():
                actor_target_avg_update[name] += delta * (num_samples /
                                                 self.total_samples)

            for name, delta in update_from_critic_target.items():
                critic_target_avg_update[name] += delta * (num_samples /
                                                  self.total_samples)                                      
        
            # Yield to other tasks in the server
            await asyncio.sleep(0)
        
        #print("line 78 ->exiting server??")
        return actor_avg_update, critic_avg_update, actor_target_avg_update, critic_target_avg_update

    
