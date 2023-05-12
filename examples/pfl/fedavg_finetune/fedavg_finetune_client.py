"""
A personalized federated learning client using FedAvg.

"""
import logging


from examples.pfl.bases import simple_personalized
from plato.config import Config
from plato.utils import fonts


class Client(simple_personalized.Client):
    """A personalized federated learning trainer using the FedAvg algorithm."""

    def load_personalized_model(self) -> None:
        """Load the personalized model.

        Each client of FedAvg will directly utilize the recevied global model as the
        personalized model.
        """
        logging.info(
            fonts.colourize(
                "[Client #%d] assings the received model [%s] to personalized model [%s].",
                colour="blue",
            ),
            self.client_id,
            Config().trainer.model_name,
            Config().trainer.personalized_model_name,
        )

        # load the received model to be personalized model
        self.trainer.personalized_model.load_state_dict(
            self.trainer.model.state_dict(), strict=True
        )
