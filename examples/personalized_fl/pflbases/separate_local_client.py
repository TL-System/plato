"""
An enhanced client for personalized federated learning.

This class enables the local model of each client to be separate from
the received global model. Doing so provides more flexibility, such
as defining the local as A+B while the model sent to the server is A. 

To implement, each client will define a unique local model that is
maintained locally. 

Following the previous example, the local model of each client is A + B.
When the model exchanged between the clients and the server is B, each client 
who receives the B will first load its local model B into A to produce the 
model A + B to be trained subsequently.
"""

import os


from pflbases import personalized_client
from pflbases import checkpoint_operator


class Client(personalized_client.Client):
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
            personalized_model=personalized_model,
        )

        # the path of the initial local model of this client
        self.init_local_mode_path = None

    def create_initial_local_model(self):
        """Create the initial local model once there is not one."""

        # get the initial local model path
        self.init_local_mode_path = self.get_init_model_path(
            model_name=self.trainer.model_name,
            prefix=self.trainer.local_model_prefix,
        )

        if not self.exist_init_local_model():
            self.trainer.define_local_model(custom_model=self.custom_model)
            self.trainer.save_model(
                filename=os.path.basename(self.init_local_mode_path),
                location=self.trainer.get_checkpoint_dir_path(),
            )

    def configure(self) -> None:
        """Further define the local model for each client."""
        super().configure()

        self.create_initial_local_model()

    def inbound_received(self, inbound_processor):
        """Reloading the local model for this client."""
        super().inbound_received(inbound_processor)

        # load the local model
        self.get_local_model()

    def get_local_model(self):
        """Getting the saved local model.

        After the local update, each client will save the local
        model (i.e., the updated global model) to the disk.
        This function is to get the saved local model.
        """

        # always get the latest local model.
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

        filename, is_searched = checkpoint_operator.search_checkpoint_file(
            filename=filename,
            checkpoints_dir=save_location,
            key_words=[model_name, prefix],
            anchor_metric="round",
            mask_words=["epoch"],
            use_latest=True,
        )

        if is_searched:
            self.trainer.load_model(filename, location=location)
        else:
            self.trainer.load_model(
                filename=os.path.basename(self.init_personalized_model_path),
                location=location,
            )

    def exist_init_local_model(self):
        """Whether this client is unselected on."""

        return os.path.exists(self.init_local_mode_path)
