"""
The implementation of client for SimCLR method.

"""

import logging

from lightly.transforms.simclr_transform import SimCLRTransform

from plato.datasources import registry as datasources_registry
from plato.clients import simple_personalized
from plato.config import Config


class Client(simple_personalized.Client):
    """The client for SimCLR approach."""

    def _load_data(self) -> None:
        """Generates data and loads them onto this client."""

        input_size = Config().trainer.input_size
        train_transform = SimCLRTransform(input_size=input_size)

        # The only case where Config().data.reload_data is set to true is
        # when clients with different client IDs need to load from different datasets,
        # such as in the pre-partitioned Federated EMNIST dataset. We do not support
        # reloading data from a custom datasource at this time.
        if (
            self.datasource is None
            or hasattr(Config().data, "reload_data")
            and Config().data.reload_data
        ):
            logging.info("[%s] Loading its data source...", self)

            if self.custom_datasource is None:
                self.datasource = datasources_registry.get(
                    client_id=self.client_id, train_transform=train_transform
                )
            elif self.custom_datasource is not None:
                self.datasource = self.custom_datasource(
                    train_transform=train_transform
                )

            logging.info(
                "[%s] Dataset size: %s with SimCLR transforme",
                self,
                self.datasource.num_train_examples(),
            )
