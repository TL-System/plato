import os
import asyncio
import logging
import websockets
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
        super().__init__(model, datasource, algorithm, trainer)

    async def start_client(self) -> None:
        """Startup function for a client."""

        if hasattr(Config().algorithm,
                   'cross_silo') and not Config().is_edge_server():
            # Contact one of the edge servers
            edge_server_id = int(Config().clients.total_clients) + (int(
                self.client_id) - 1) % int(Config().algorithm.total_silos) + 1
            logging.info("[Client #%d] Contacting Edge server #%d.",
                         self.client_id, edge_server_id)

            assert hasattr(Config().algorithm, 'total_silos')

            uri = 'ws://{}:{}'.format(
                Config().server.address,
                int(Config().server.port) + edge_server_id)
        else:
            logging.info("[Client #%d] Contacting the central server.",
                         self.client_id)
            uri = 'ws://{}:{}'.format(Config().server.address,
                                      Config().server.port)

        try:
            old_weights = None
            async with websockets.connect(uri,
                                          ping_interval=None,
                                          max_size=2**30) as websocket:
                logging.info("[Client #%d] Signing in at the server.",
                             self.client_id)
                await websocket.send(pickle.dumps({'id': self.client_id}))

                while True:
                    logging.info("[Client #%d] Waiting to be selected.",
                                 self.client_id)
                    server_response = await websocket.recv()
                    data = pickle.loads(server_response)

                    if data['id'] == self.client_id:
                        self.process_server_response(data)
                        logging.info("[Client #%d] Selected by the server.",
                                     self.client_id)

                        if not self.data_loaded:
                            self.load_data()

                        if 'payload' in data:
                            server_payload = await self.recv(
                                self.client_id, data, websocket)
                            self.load_payload(server_payload)
                            # get old weights
                            old_weights = server_payload

                        self.client_id = int(self.client_id)
                        report, payload = await self.train()
                        report.belive = self.cos_feature_distance(
                            old_weights,
                            payload)  # self.sampler.trainset_size()
                        self.client_id = str(self.client_id)

                        if Config().is_edge_server():
                            logging.info(
                                "[Server #%d] Model aggregated on edge server (client #%d).",
                                os.getpid(), self.client_id)
                        else:
                            logging.info("[Client #%d] Model trained.",
                                         self.client_id)

                        # Sending the client report as metadata to the server (payload to follow)
                        client_report = {
                            'id': self.client_id,
                            'report': report,
                            'payload': True
                        }
                        await websocket.send(pickle.dumps(client_report))

                        # Sending the client training payload to the server
                        await self.send(websocket, payload)

        except OSError as exception:
            logging.info("[Client #%d] Connection to the server failed.",
                         self.client_id)
            logging.error(exception)

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
    asyncio.run(client.start_client())


if __name__ == "__main__":
    main()
