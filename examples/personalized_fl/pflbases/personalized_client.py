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

from plato.clients import simple
from plato.config import Config
from pflbases.filename_formatter import NameFormatter

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
        # this object personalized model is a class but not an instance
        self.custom_personalized_model = personalized_model
        self.personalized_model = None

        # the path of the initial personalized model of this client
        self.init_personalized_model_path = None

    def configure(self) -> None:
        """Performing the general client's configure and then initialize the
        personalized model for the client."""
        super().configure()

        # jump out if no personalization info is provided
        if not hasattr(Config().algorithm, "personalization"):
            sys.exit(
                "Error: personalization block must be provided under the algorithm."
            )

        # set the personalized model class
        if (
            self.personalized_model is None
            and self.custom_personalized_model is not None
        ):
            self.personalized_model = self.custom_personalized_model

        # set the indicators for personalization of the trainer
        self.trainer.do_round_personalization = self.is_round_personalization()
        self.trainer.do_final_personalization = self.is_final_personalization()

        # get the initial personalized model path
        self.init_personalized_model_path = self.get_init_model_path(
            model_name=self.trainer.personalized_model_name,
            prefix=self.trainer.personalized_model_prefix,
        )

        # if this client does not its initialized personalized model yet.
        # and the personalized model is required in the subsequent learning
        if (
            not self.exist_init_personalized_model()
            and self.is_personalized_model_required()
        ):
            # define its personalized model if it is not defined yet
            if self.trainer.personalized_model is None:
                self.trainer.define_personalized_model(self.personalized_model)
            else:
                # only reinitialize the model based on the client id
                # as the random seed
                self.trainer.reinitialize_personalized_model()

            self.trainer.save_personalized_model(
                filename=os.path.basename(self.init_personalized_model_path),
                location=self.trainer.get_checkpoint_dir_path(),
            )

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model for this client before any operations."""
        super().inbound_received(inbound_processor)

        # load the personalized model when
        # 1. the personalization is performed per round
        # 2. the current round is larger than the total rounds,
        #   which means the final personalization.
        if self.is_personalized_model_required():
            self.get_personalized_model()

    def get_personalized_model(self):
        """Getting the personalized model of the client.
        Default, the latest personalized model is obtained.
        """
        # always get the latest personalized model.
        desired_round = self.current_round - 1
        model_name = self.trainer.personalized_model_name
        prefix = self.trainer.personalized_model_prefix
        save_location, filename = self.trainer.get_model_checkpoint_path(
            model_name=model_name,
            prefix=prefix,
            round_n=desired_round,
            epoch_n=None,
        )

        filename, is_searched = checkpoint_operator.search_checkpoint_file(
            filename=filename,
            checkpoints_dir=save_location,
            key_words=[model_name, prefix],
            anchor_metric="round",
            mask_words=["epoch"],
            use_latest=True,
        )
        if is_searched:
            self.trainer.load_personalized_model(filename, location=save_location)
        else:
            self.trainer.load_personalized_model(
                filename=os.path.basename(self.init_personalized_model_path),
                location=save_location,
            )

    def get_init_model_path(self, model_name: str, prefix: str):
        """Get the path of saved initial model (untrained)."""
        checkpoint_dir_path = self.trainer.get_checkpoint_dir_path()

        filename = NameFormatter.get_format_name(
            model_name=model_name,
            client_id=self.client_id,
            round_n=0,
            epoch_n=None,
            run_id=None,
            prefix=prefix,
            ext="pth",
        )
        model_path = os.path.join(checkpoint_dir_path, filename)

        return model_path

    def exist_init_personalized_model(self):
        """Whether this client is unselected on."""

        return os.path.exists(self.init_personalized_model_path)

    def is_personalized_model_required(self):
        """whether a well-defined personalized model is required."""
        return (
            self.trainer.do_round_personalization
            or self.trainer.do_final_personalization
        )

    def is_final_personalization(self):
        """Get whether the client is performing the final personalization.
        whether the client is performing the final personalization
        the final personalization is mandatory
        """
        if self.current_round > Config().trainer.rounds:
            return True
        return False

    def is_round_personalization(self):
        """Get whether the client is performing the round personalization.
        whether the client is perfomring the personalization in the current round
        this round personalization should be determined by the user
        depending on the algorithm.
        """

        if (
            hasattr(Config().algorithm.personalization, "do_personalization_per_round")
            and Config().algorithm.personalization.do_personalization_per_round
        ):
            return True
        return False
