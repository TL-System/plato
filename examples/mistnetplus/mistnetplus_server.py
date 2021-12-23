import logging
import os
import sys
import pickle
from dataclasses import dataclass

import torch

os.environ['config_file'] = 'examples/mistnetplus/mistnet_lenet5_server.yml'

from plato.config import Config
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.servers import fedavg

import split_learning_algorithm
import split_learning_trainer


@dataclass
class Report:
    """Client report sent to the MistNet federated learning server."""
    num_samples: int
    payload_length: int
    phase: str


class MistnetplusServer(fedavg.Server):
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

    async def client_payload_done(self, sid, client_id, s3_key=None):
        if s3_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(s3_key)
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        # if clients send features, train it and return gradient
        if self.reports[sid].phase == "features":
            logging.info(
                "[Server #%d] client #%d features received. Processing.",
                os.getpid(), client_id)
            features = [self.client_payload[sid]]
            feature_dataset = feature.DataSource(features)
            sampler = all_inclusive.Sampler(feature_dataset)
            self.algorithm.train(feature_dataset, sampler,
                                 Config().algorithm.cut_layer)
            # Test the updated model
            self.accuracy = self.trainer.test(self.testset)
            logging.info(
                '[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
                    os.getpid(), 100 * self.accuracy))

            payload = self.load_gradients()
            logging.info("[Server #%d] Reporting gradients to client #%d.",
                         os.getpid(), client_id)

            sid = self.clients[client_id]['sid']
            # payload = await self.customize_server_payload(pickle.dumps(payload))
            # Sending the server payload to the clients
            payload = self.load_gradients()
            await self.send(sid, payload, client_id)
            return

        self.updates.append((self.reports[sid], self.client_payload[sid]))

        if len(self.updates) > 0 and len(self.updates) >= len(
                self.selected_clients):
            logging.info(
                "[Server #%d] All %d client reports received. Processing.",
                os.getpid(), len(self.updates))
            await self.process_reports()
            await self.wrap_up()
            await self.select_clients()

    async def aggregate_weights(self, updates):
        model = self.algorithm.extract_weights()
        update = await self.federated_averaging(updates)
        feature_update = self.algorithm.update_weights(update)

        for name, weight in model.items():
            if name == Config().algorithm.cut_layer:
                logging.info("[Server #%d] %s cut", os.getpid(), name)
                break
            model[name] = model[name] + feature_update[name]

        self.algorithm.load_weights(model)

    def load_gradients(self):
        """ Loading gradients from a file. """
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        model_path = f'{model_dir}{model_name}_gradients.pth'
        logging.info("[Server #%d] Loading gradients from %s.", os.getpid(),
                     model_path)

        return torch.load(model_path)


def main():
    """A Plato federated learning training session using a custom model. """
    trainer = split_learning_trainer.Trainer()
    algorithm = split_learning_algorithm.Algorithm(trainer=trainer)
    server = MistnetplusServer(algorithm=algorithm, trainer=trainer)
    server.run()


if __name__ == "__main__":
    main()
