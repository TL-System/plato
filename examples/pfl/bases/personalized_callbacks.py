"""
The processors for loading, assigning, and adjusting the models for local update.

Note:
    Plato simulates the client with multiple `processes`, which are started at the 
    beginning of running. These The server will build fake connection with these `processes` 
    - when the `processes` receive information from the server, the `configuration()` 
    function is called the first time to perform necessary initialization 
    (model, trainer, algorithm, personalized model). Server will solely select clients by 
    sampling ids, which will be assigned to these `processes` to simulate that some clients 
    are actually selected by the server. However, the model and any data of `processes` are 
    reminded unchanged/outdated, thus makes no sense to those clients. For example, in round 
    `r`, the first `process` is assigned with client id 10 to simulate the selection of 
    client 10. But the model parameters and learning status of this `process` belong to the 
    local update process of client 2 who are selected and simulated by this `process` in round
    `r-1`.
    
    Therefore, only when each "client"/`process` receives payload holding model parameters from 
    the server, the model and learning status of this "client"/`process` become meaningful to this
    client becasue these terms are updated by the payload for current round.

    This insight leads to a core conclusion:
    Any membership variables of the client class should be updated in each round to hold this client's
    own status. For example, in the personalized learning or scenarios containing local model, the client
    should load its own previously saved status to refresh variables.

"""

import logging
from typing import Any

from plato.processors import base
from plato.callbacks import client
from plato.algorithms import fedavg_partial


class ModelStatusProcessor(base.Processor):
    """
    A client processor used to reload the model status for current client.
    """

    def process(self, data: Any) -> Any:
        logging.info(
            "[Client #%d] Received the payload containing modules: %s.",
            self.trainer.client_id,
            fedavg_partial.Algorithm.extract_modules_name(list(data.keys())),
        )

        return data


class ClientCallback(client.ClientCallback):
    def on_inbound_received(self, client, inbound_processor):
        """
        Event called before inbound processors start to process data.

        Each client will process the payload data based on its
        personal requirements.
        """

        # reload the personalized model saved locally from
        # previous round
        inbound_processor.processors.append(
            ModelStatusProcessor(
                trainer=client.personalized_trainer,
            )
        )
