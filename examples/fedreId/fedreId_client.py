import os
import asyncio
import logging
import pickle

import torch
from torch import nn
import torch.nn.functional as F

os.environ['config_file'] = 'examples/configs/client.yml'
from fedreId import DataSource, Trainer
from plato.clients import simple
from plato.config import Config

class fedReIdClient(simple.Client):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model=model, datasource=datasource, trainer=trainer)
    
    async def train(self):
        old_weights = self.algorithm.extract_weights()
        report, weights = await super().train()
        belive = self.cos_feature_distance(old_weights, weights)
        return report, [weights, belive]

    def cos_feature_distance(self, old_weights, new_weights):
        if old_weights == None:
            logging.info("old_weights is None")
            return self.sampler.trainset_size()
        dis = []
        # option II
        # self.load_payload(old_weights)
        # old_feature = self.trainer.test_output(Config().trainer._asdict(), self.datasource.get_test_set())
        # self.load_payload(new_weights)
        # new_feature = self.trainer.test_output(Config().trainer._asdict(), self.datasource.get_test_set())

        # for i in range(len(old_feature)):
        #     # print(old_feature[i].shape, new_feature[i].shape)
        #     distance = 1.0 - F.cosine_similarity(old_feature[i].float(), new_feature[i].float(), 0)
        #     dis.append(distance)

        for i in old_weights:
            distance = 1.0 - F.cosine_similarity(old_weights[i].float(),
                                                 new_weights[i].float(), 0)
            dis.append(torch.mean(distance))

        print(dis)
        return sum(dis) / len(dis)
    
def main():
    """A Plato federated learning training session using a custom client. """
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    datasource = DataSource()
    trainer = Trainer(model=model)
    client = fedReIdClient(model=model, datasource=datasource, trainer=trainer)
    client.configure()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start_client())

if __name__ == "__main__":
    main()
