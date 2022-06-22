"""
A federated learning client for td3.
"""
import logging
from dataclasses import dataclass
from plato.clients import simple
import numpy as np
import pybullet_envs
from plato.config import Config

@dataclass
class Report(simple.Report):
    """A client report to be sent to the federated learning server."""
    client_id: int
    actor_loss: float
    critic_loss: float


class RLClient(simple.Client):
    
    def __init__(self, trainer = None, model = None, algorithm = None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        logging.info("A custom client has been initialized!")
        

    async def train(self):
        """The machine learning training workload on a client."""
        report, weights = await super().train()

        actor_loss = self.get_loss(True)
        critic_loss = self.get_loss(False)

        print("========")
        print(actor_loss)
        print("========")
        print(critic_loss)
        
        
        return Report(report.num_samples, report.accuracy, report.training_time, \
         report.comm_time, report.update_response, self.client_id, actor_loss, critic_loss), weights

    def get_loss(self, actor_passed):
         loss = self.trainer.load_loss(actor_passed)
         return loss




