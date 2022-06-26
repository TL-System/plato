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
    entropy_loss: float
    actor_grad: float
    critic_grad: float
    actor_fisher: float
    critic_fisher: float


class RLClient(simple.Client):
    
    def __init__(self, trainer = None, model = None, algorithm = None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        logging.info("A custom client has been initialized!")
        

    async def train(self):
        """The machine learning training workload on a client."""
        report, weights = await super().train()

        actor_loss, critic_loss, entropy_loss = self.get_loss()
        actor_grad, critic_grad = self.get_grad()
        actor_fisher, critic_fisher = self.get_fisher()
        print("In client fisher", actor_fisher)
        
        return Report(report.num_samples, report.accuracy, report.training_time, \
         report.comm_time, report.update_response, self.client_id, actor_loss, critic_loss, \
          entropy_loss, actor_grad, critic_grad, actor_fisher, critic_fisher), weights

    def get_loss(self):
        return self.trainer.load_loss()

    def get_grad(self):
        return self.trainer.load_grads()

    def get_fisher(self): 
        return self.trainer.load_fisher()




