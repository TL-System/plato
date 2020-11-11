"""
A basic federated learning institution who receives weights updates from its clients
and sends aggregated weight updates to the server.
"""

import logging
import json
import pickle
import websockets

from servers import FedAvgServer
from models import registry as models_registry


class SimpleInstitution(FedAvgServer):
    """A basic federated learning institution."""

    def __init__(self, config, institution_id):
        super().__init__(config)
        self.institution_id = institution_id
        self.do_test = None # Should the institution test the trained model?
        self.testset = None # Testing dataset
        self.report = None # Report to the server
        self.model = None # Machine learning model
        self.aggregations = None # The number of aggregations on this institution in each global training round
        
    async def start_institution(self):
        uri = 'ws://{}:{}'.format(self.config.server.address, self.config.server.port)

        async with websockets.connect(uri, max_size=2 ** 30) as websocket:
            logging.info("Signing in at the server with institution ID %s...", self.institution_id)
            await websocket.send(json.dumps({'id': self.institution_id}))

            while True:
                logging.info("Waiting for the server to start training...")
                server_response = await websocket.recv()
                data = json.loads(server_response)

                if data['id'] == self.institution_id and 'payload' in data:
                    logging.info("Training starts -- receiving the model...")
                    server_model = await websocket.recv()
                    self.model.load_state_dict(pickle.loads(server_model))

                    # Run for one aggregation
                    await self.one_aggregation(websocket)

                    # Aggregete reports from selected clients


                    logging.info("Model aggregated on institution #%s...", self.institution_id)
                    # Sending institution ID as metadata to the server (payload to follow)
                    institution_update = {'id': self.institution_id, 'payload': True}
                    await websocket.send(json.dumps(institution_update))

                    # Sending the report to the server as payload
                    await websocket.send(pickle.dumps(self.report))


    def configure(self):
        """Prepare this institution for training."""
        self.aggregations = self.config.institutions.aggregations
 
        model_name = self.config.training.model
        self.model = models_registry.get(model_name, self.config)


    async def one_aggregation(self, websocket):
        """
        Selecting some clients of this institution to participate in the current aggregation round,
        and run them for one aggregation round.
        """
        selected_clients = self.select_clients()
        self.reports = []

        assert len(selected_clients) > 0



class Report:
    """Federated learning institution report."""

    def __init__(self, institution, weights):
        self.institution_id = institution.institution_id
        self.num_samples = 0
        self.weights = weights
        self.accuracy = 0

    def set_accuracy(self, accuracy):
        """Include the test accuracy computed at an institution in the report."""
        self.accuracy = accuracy