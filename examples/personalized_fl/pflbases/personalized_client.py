"""
A basic personalized federated learning client who performs the 
global learning and local learning.

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
    the server, the model of this "client"/`process` become meaningful to this
    client becasue this term will be updated by the payload for current round.

    This insight leads to a core conclusion:
    Any membership variables of the client class should be updated in each round to hold this client's
    own status. For example, in the personalized learning or scenarios containing local model, the client
    should load its own previously saved status to refresh variables.

"""
import sys
import os
import logging

from plato.clients import simple
from plato.config import Config
from plato.utils import fonts
from plato.utils.filename_formatter import NameFormatter

from pflbases import checkpoint_operator


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
        personalized_model=None,
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

        # the personalized model here corresponds to the client's
        # personal needs.
        self.custom_personalized_model = personalized_model
        self.personalized_model = None

        # the path of the initial personalized model of this client
        self.init_personalized_model_path = None

        # whether this client contains the corresponding
        # personalized model
        self.new_client = False

    def configure(self) -> None:
        """Performing the general client's configure and then initialize the
        personalized model for the client."""
        super().configure()

        # jump out if no personalized model is set
        if not hasattr(Config().algorithm, "personalization"):
            sys.exit(
                "Error: personalization block must be provided under the algorithm."
            )

        # define the personalized model
        if (
            self.personalized_model is None
            and self.custom_personalized_model is not None
        ):
            self.personalized_model = self.custom_personalized_model

        # get the initial personalized model path
        self.init_personalized_model_path = self.get_init_personalized_model_path()

        # if this client does not a personalized model yet.
        # define a an initial one and save to the disk
        if not self.exist_init_personalized_model():
            # define its personalized model
            self.trainer.define_personalized_model(self.personalized_model)
            self.trainer.save_personalized_model(
                filename=os.path.basename(self.init_personalized_model_path),
                location=self.trainer.get_checkpoint_dir_path(),
            )
        self.personalized_model = self.trainer.personalized_model

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model for this client before any operations."""
        super().inbound_received(inbound_processor)

        # load the personalized model when
        # 1. the personalization is performed per round
        # 2. the current round is larger than the total rounds,
        #   which means the final personalization.
        if (
            self.train.is_round_personalization()
            or self.trainer.is_final_personalization()
        ):
            self.get_personalized_model()

    def get_personalized_model(self):
        """Getting the personalized model of the client."""

        # always get the latest personalized model.
        desired_round = self.current_round - 1
        location = self.get_checkpoint_dir_path()

        filename, is_searched = checkpoint_operator.search_client_checkpoint(
            client_id=self.client_id,
            checkpoints_dir=location,
            model_name=self.trainer.personalized_model_name,
            current_round=desired_round,
            run_id=None,
            epoch=None,
            prefix=self.trainer.personalized_model_checkpoint_prefix,
            anchor_metric="round",
            mask_words=["epoch"],
            use_latest=True,
        )
        if is_searched:
            self.trainer.load_personalized_model(filename, location=location)
        else:
            self.trainer.load_personalized_model(
                filename=os.path.basename(self.init_personalized_model_path),
                location=self.trainer.get_checkpoint_dir_path(),
            )

    def get_init_personalized_model_path(self):
        """Get the path of the personalized model."""
        checkpoint_dir_path = self.trainer.get_checkpoint_dir_path()

        filename = NameFormatter.get_format_name(
            model_name=self.trainer.personalized_model_name,
            client_id=self.client_id,
            round_n=0,
            epoch_n=None,
            run_id=None,
            prefix=self.trainer.personalized_model_checkpoint_prefix,
            ext="pth",
        )
        model_path = os.path.join(checkpoint_dir_path, filename)

        return model_path

    def exist_init_personalized_model(self):
        """Whether this client is unselected on."""

        return not os.path.exists(self.init_personalized_model_path)
