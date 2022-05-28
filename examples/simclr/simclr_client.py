"""
Implement the client for the SimCLR method.

"""

import logging

from plato.config import Config
from plato.clients import simple
from plato.datasources import registry as datasources_registry
from plato.datasources import datawrapper_registry
from plato.samplers import registry as samplers_registry
from plato.datasources.augmentations import get_aug


class Client(simple.Client):

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

        # using the name memory is general in this domain,
        #   it aims to record the train loader without using
        #   the data augmentation.
        # thus, it utilizes the same transform as the test loader
        #   for monitor (i.e., contrastive learning monitor)
        #   but performs on the train dataset.
        # for this trainset loader,
        #   - train: True, as it utilize the trainset
        #   - shuffle: False
        #   - transform: set the 'train' to be False to
        #       use the general transform,
        #       i.e., eval_aug under 'datasource/augmentations/eval_aug.py'
        self.memory_trainset = None

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        super().load_data()

        if hasattr(Config().data,
                   "data_wrapper") and Config().data.data_wrapper != None:
            augment_transformer_name = None

            if hasattr(Config().data, "augment_transformer_name"
                       ) and Config().data.augment_transformer_name != None:
                image_size = Config().trainer.image_size
                augment_transformer_name = Config(
                ).data.augment_transformer_name
                augment_transformer = get_aug(name=augment_transformer_name,
                                              image_size=image_size,
                                              train=True,
                                              train_classifier=None)

            self.trainset = datawrapper_registry.get(self.trainset,
                                                     augment_transformer)

            general_augment_transformer = get_aug(name="general_transform",
                                                  image_size=image_size,
                                                  train=False,
                                                  train_classifier=False)
            # get the same trainset again for memory trainset
            self.memory_trainset = self.datasource.get_train_set()
            self.memory_trainset = datawrapper_registry.get(
                self.memory_trainset, general_augment_transformer)

        if Config().clients.do_test:
            if hasattr(Config().data,
                       "data_wrapper") and Config().data.data_wrapper != None:
                augment_transformer = None

                if hasattr(Config().data, "augment_transformer"
                           ) and Config().data.augment_transformer != None:
                    image_size = Config().trainer.image_size

                    augment_transformer = get_aug(name="general_transform",
                                                  image_size=image_size,
                                                  train=False,
                                                  train_classifier=None)

                self.testset = datawrapper_registry.get(
                    self.testset, augment_transformer)
