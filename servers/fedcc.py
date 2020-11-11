"""
A simple cross-silo federated learning server using federated averaging.
"""

import logging
import subprocess
import pickle
import json

from institutions import SimpleInstitution
from training import trainer
from servers import FedAvgServer


class CrossSiloServer(FedAvgServer):
    """Cross-silo federated learning server using federated averaging."""

    def __init__(self, config):
        super().__init__(config)
        self.institutions = {}
        self.loader = None


    def register_institution(self, institution_id, websocket):
        if not institution_id in self.institutions:
            self.institutions[institution_id] = websocket

        logging.info("institutions: %s", self.institutions)


    def unregister_institution(self, websocket):
        for key, value in dict(self.institutions).items():
            if value == websocket:
                del self.institutions[key]

        logging.info("institutions: %s", self.institutions)


    def start_institutions(self):
        """Starting all the institutions as separate processes."""
        for institution_id in range(1, self.config.institutions.total + 1):
            logging.info("Starting institution #%s...", institution_id)
            command = "python institution.py -i {}".format(institution_id)
            subprocess.Popen(command, shell=True)

    
    async def wait_for_institutions(self, websocket):
        """Waiting for institutions to arrive."""
        async for message in websocket:
            data = json.loads(message)
            institution_id = data['id']
            logging.info("New institution arrived with ID: %s", institution_id)

            # a new institution arrives
            assert 'payload' not in data
            self.register_institution(institution_id, websocket)

            if len(self.institutions) == self.config.institutions.total:
                return


    async def one_round(self, websocket):
        self.reports = []

        for institution_id in self.institutions:
            socket = self.institutions[institution_id]
            logging.info("Institution #%s...", institution_id)
            server_response = {'id': institution_id, 'payload': True}
            await socket.send(json.dumps(server_response))

            logging.info("Sending the current model...")
            await socket.send(pickle.dumps(self.model.state_dict()))

        async for message in websocket:
            data = json.loads(message)
            institution_id = data['id']
            logging.info("institution data received with ID: %s", institution_id)

            if 'payload' in data:
                # an existing institution reports new updates from local training
                institution_update = await websocket.recv()
                report = pickle.loads(institution_update)
                logging.info("Institution update received. Accuracy = {:.2f}%\n"
                    .format(100 * report.accuracy))

                self.reports.append(report)

                if len(self.reports) == len(self.institutions):
                    return self.process_report()
            else:
                # a new institution arrives
                self.register_institution(institution_id, websocket)


    def process_report(self):
        updated_weights = self.aggregate_weights(self.reports)
        trainer.load_weights(self.model, updated_weights)

        # Test the global model accuracy
        if self.config.institutions.do_test:  # Get average accuracy from institution reports
            accuracy = self.accuracy_averaging(self.reports)
            logging.info('Average institution accuracy: {:.2f}%\n'.format(100 * accuracy))
        else: # Test the updated model on the server
            testset = self.loader.get_testset()
            batch_size = self.config.training.batch_size
            testloader = trainer.get_testloader(testset, batch_size)
            accuracy = trainer.test(self.model, testloader)
            logging.info('Global model accuracy: {:.2f}%\n'.format(100 * accuracy))

        return accuracy


