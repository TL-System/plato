"""
A basic personalized federated learning client who performs the 
global learning and local learning.

Note:
    Plato simulates the client with multiple `processes`, which are started at the 
    beginning of running. The server will build fake connection with these `processes` 
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
    the server, the model of this "client"/`process` become meaningful to this
    client becasue this term will be updated by the payload for current round.

    This insight leads to a core conclusion:
    Any membership variables of the client class should be updated in each round to hold this client's
    own status. For example, in the personalized learning or scenarios containing local model, the client
    should load its own previously saved status to refresh variables.

"""
import sys
import os

from plato.clients import simple
from plato.config import Config
from pflbases.filename_formatter import NameFormatter

from pflbases import trainer_utils


class Client(simple.Client):
    """A basic personalized federated learning client."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )

        # The path of the initial local model of this client.
        self.init_local_model_path = None

    def configure(self) -> None:
        """Performing the general client's configure and then initialize the
        local model for the client."""
        super().configure()

        # Jump out if no personalization info is provided.
        if not hasattr(Config().algorithm, "personalization"):
            sys.exit(
                "Error: personalization block must be provided under the algorithm."
            )

        # Create the initial local model for this client.
        self.create_initial_local_model()

    def create_initial_local_model(self):
        """Creating the initial local modle for this client."""

        # Get the initial local model path
        self.init_local_model_path = self.get_init_model_path(
            model_name=self.trainer.model_name,
            prefix=self.trainer.local_model_prefix,
        )

        # If this client have not initialized its personalized model yet
        # and the personalized model is required in the subsequent learning.
        if not self.exist_init_local_model():
            # Only reinitialize the model based on the client id
            # as the random seed.
            self.trainer.reinitialize_local_model()
            self.trainer.save_model(
                filename=os.path.basename(self.init_local_model_path),
                location=self.trainer.get_checkpoint_dir_path(),
            )

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model for this client before any operations."""
        super().inbound_received(inbound_processor)

        # Get the local model
        self.get_local_model()

    def get_local_model(self):
        """
        Getting the local model of the client.
        Default, the latest model of the client is obtained.
        """
        # Always get the latest local model.
        desired_round = self.current_round - 1
        location = self.trainer.get_checkpoint_dir_path()
        model_name = self.trainer.model_name
        prefix = self.trainer.local_model_prefix
        save_location, filename = self.trainer.get_model_checkpoint_path(
            model_name=model_name,
            prefix=prefix,
            round_n=desired_round,
            epoch_n=None,
        )

        filename, is_searched = trainer_utils.search_checkpoint_file(
            checkpoint_dir=save_location,
            filename=filename,
            key_words=[model_name, prefix],
            anchor_metric="round",
            mask_words=["epoch"],
            use_latest=True,
        )

        if is_searched:
            self.trainer.load_model(filename, location=location)
        else:
            self.trainer.load_model(
                filename=os.path.basename(self.init_local_model_path),
                location=location,
            )

    def get_init_model_path(self, model_name: str, prefix: str):
        """Get the path of saved initial model (untrained)."""

        checkpoint_dir_path = self.trainer.get_checkpoint_dir_path()
        filename = NameFormatter.get_format_name(
            model_name=model_name,
            client_id=self.client_id,
            round_n=0,
            epoch_n=None,
            prefix=prefix,
            ext="pth",
        )
        model_path = os.path.join(checkpoint_dir_path, filename)

        return model_path

    def exist_init_local_model(self):
        """Whether this client is unselected on."""

        return os.path.exists(self.init_local_model_path)
