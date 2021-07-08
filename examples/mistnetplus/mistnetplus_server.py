import logging
import os
from itertools import chain

from plato.config import Config
from plato.samplers import all_inclusive
from plato.servers import fedavg
from mistnetplus import DataSource, Trainer

class Server(fedavg.Server):
    """The split learning server."""
    def __init__(self):
        super().__init__()
        assert Config().trainer.rounds == 1

    async def process_reports(self):
        """Process the features extracted by the client and perform server-side training."""
        features = [features for (__, features) in self.updates]

        # Faster way to deep flatten a list of lists compared to list comprehension
        feature_dataset = list(chain.from_iterable(features))

        # Training the model using all the features received from the client
        sampler = all_inclusive.Sampler(feature_dataset)
        self.algorithm.train(feature_dataset, sampler,
                             Config().algorithm.cut_layer)

        # Test the updated model
        self.accuracy = self.trainer.test(self.testset)
        logging.info('[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
            os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()

    async def wrap_up(self):
        """ Wrapping up when each round of training is done. """
        # Report gradients to client
        payload = self.load_gradients()
        self.updates = []
        # if len(payload) > 0:
        for client_id in self.selected_clients:
            logging.info("[Server #%d] Reporting gradients to client #%d.",
                         os.getpid(), client_id)
            server_response = {
                'id': client_id,
                'payload': True,
                'payload_length': len(payload)
            }
            server_response = await self.customize_server_response(
                server_response)
            # Sending the server response as metadata to the clients (payload to follow)
            actual_client_id = [k for k,v in self.clients.items() if v['virtual_id'] == client_id][0]
            socket = self.clients[actual_client_id]
            await socket.send(pickle.dumps(server_response))

            payload = await self.customize_server_payload(payload)

            # Sending the server payload to the clients
            await self.send(socket, payload)

            # Wait until client finish its train
            report = await self.clients[actual_client_id].recv()
            payload = await self.clients[actual_client_id].recv()
            self.updates.append(report, payload)
            
        # do_avg
        after_model = self.algorithm.extract_weights()
        await self.aggregate_weights(self.updates)
        before_model = self.algorithm.extract_weights()
        
        final_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in final_update.items()
        }
        layers = 0
        for name, weight in before_model.item():
            if layers <= Config().algorithm.cut_layer:
                final_update[name] = before_model[name]
            else:
                final_update[name] = final_update[name]
        
        # Break the loop when the target accuracy is achieved
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy and self.accuracy >= target_accuracy:
            logging.info("[Server #%d] Target accuracy reached.", os.getpid())
            await self.close()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")
            await self.close()

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
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    server = fedReIdServer(model=model)
    server.run()


if __name__ == "__main__":
    main()

