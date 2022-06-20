"""
A federated learning client for td3.
"""
import logging
from dataclasses import dataclass
from plato.clients import simple
import numpy as np
import pybullet_envs

@dataclass
class Report(simple.Report):
    """A client report to be sent to the federated learning server."""
    client_id: int
    #average_reward: int


class RLClient(simple.Client):
    
    def __init__(self, trainer = None, model = None, algorithm = None):
        #print("are we even ever in this consturctor line 30 of td_client")
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        logging.info("A custom client has been initialized!")

    async def train(self):
         
        report, weights = await super().train()
      
        return Report(report.num_samples, report.accuracy, report.training_time, \
         report.comm_time, report.update_response, self.client_id), weights


