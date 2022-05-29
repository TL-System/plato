"""
Implement the client for the SimCLR method.

"""

import logging
import time
from dataclasses import dataclass

from plato.config import Config
from plato.clients import simple
from plato.clients import base
from plato.datasources import registry as datasources_registry
from plato.datasources import datawrapper_registry
from plato.samplers import registry as samplers_registry
from plato.datasources.augmentations.augmentation_register import get as get_aug

from plato.utils import fonts


@dataclass
class Report(base.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    comm_time: float
    update_response: bool


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
                augment_transformer_name = Config(
                ).data.augment_transformer_name
                augment_transformer = get_aug(name=augment_transformer_name,
                                              train=True)

            self.trainset = datawrapper_registry.get(self.trainset,
                                                     augment_transformer)

            # get the same trainset again for memory trainset
            # this dataset is prepared for the representation learning
            #   monitor
            general_augment_transformer = get_aug(
                name=augment_transformer_name, train=False)

            self.memory_trainset = self.datasource.get_train_set()
            self.memory_trainset = datawrapper_registry.get(
                self.memory_trainset, general_augment_transformer)

        if Config().clients.do_test:
            if hasattr(Config().data,
                       "data_wrapper") and Config().data.data_wrapper != None:
                augment_transformer = None

                if hasattr(
                        Config().data, "augment_transformer_name"
                ) and Config().data.augment_transformer_name != None:

                    augment_transformer = get_aug(
                        name=augment_transformer_name, train=False)

                self.testset = datawrapper_registry.get(
                    self.testset, augment_transformer)

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info(
            fonts.colourize(
                f"[{self}] Started training in communication round #{self.current_round}."
            ))

        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError:
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        # This is the monitor test performed to measure the representation's quality
        # based on the cluster method, such as k-nearest neighbors (KNN). The detailed
        # procedures of this test are:
        # 1.- Extract representation from the memory trainset based on the trained
        # encoder of the self-supervised methods. The memory trainset is a duplicate
        # of the trainset but without applying the contrastive data augmentation.
        # It only utilizes the normal transform, as shown in 'datasources/augmentations/test_aug.py' .
        # 2.- Train the KNN method based on the extracted representation of the memory trainset.
        # 3.- Using the trained KNN method to classify the testset to obtain accuracy.

        if (hasattr(Config().clients, 'do_test') and Config().clients.do_test
            ) and (not hasattr(Config().clients, 'test_interval') or
                   self.current_round % Config().clients.test_interval == 0):

            accuracy = self.trainer.test(testset=self.testset,
                                         sampler=self.testset_sampler,
                                         memory_trainset=self.memory_trainset,
                                         memory_trainset_sampler=self.sampler)

            if accuracy == -1:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            if hasattr(Config().trainer, 'target_perplexity'):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self,
                             100 * accuracy)
        else:
            accuracy = 0

        # In general, the performance of self-supervised learning methods is
        # measured by applying their encoder to extract representation for the
        # downstream tasks; thus, the quantity metrics are reported based on the
        # objective of these tasks. This is commonly called linear evaluation and
        # is conducted after completing the training of self-supervised learning
        # methods. However, in federated learning, it is expected to track its
        # performance after several rounds of communication. Therefore, in every
        # #eval_test_interval round, the linear evaluation will be conducted to
        # obtain accuracy or other metrics. The procedures are:
        # 1.- Design a very simple model for the downstream task, such as the
        # image classification. The classifier can be a one layer fully-connected layer.
        # 2.- Use the encoder of the trained self-supervised method as the
        # backbone to extract representation from the train data.
        # 3.- Combine the backbone and the designed model to perform the
        # downstream task. Thus, the input samples are processed by the
        # backbone to generate representation, which is used as input for
        # the designed model to complete the task. This encoder/backbone is
        # frozen without any changes. Only the designed model is optimized.
        if (hasattr(Config().clients, 'do_test') and Config().clients.do_test
            ) and (not hasattr(Config().clients, 'eval_test_interval') or
                   self.current_round % Config().clients.test_interval == 0):

            accuracy = self.trainer.eval_test(
                testset=self.testset,
                sampler=self.testset_sampler,
                memory_trainset=self.memory_trainset,
                memory_trainset_sampler=self.sampler)

            if accuracy == -1:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            if hasattr(Config().trainer, 'target_perplexity'):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self,
                             100 * accuracy)
        else:
            accuracy = 0

        comm_time = time.time()

        if hasattr(Config().clients,
                   'sleep_simulation') and Config().clients.sleep_simulation:
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time
            self.report = Report(self.sampler.trainset_size(), accuracy,
                                 (avg_training_time + sleep_seconds) *
                                 Config().trainer.epochs, comm_time, False)
        else:
            self.report = Report(self.sampler.trainset_size(), accuracy,
                                 training_time, comm_time, False)

        return self.report, weights
