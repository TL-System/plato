"""
A basic federated self-supervised learning client
who performs the pre-train and evaluation stages locally.

"""

import logging
import time
from dataclasses import dataclass

from plato.config import Config
from plato.clients import pers_simple
from plato.clients import base
from plato.models import general_mlps_register as general_MLP_model
from plato.datasources import datawrapper_registry
from plato.samplers import registry as samplers_registry

from plato.datasources.augmentations.augmentation_register import get as get_aug

from plato.utils import fonts


@dataclass
class Report(base.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    comm_time: float
    update_response: bool


class Client(pers_simple.Client):
    """A basic self-supervised federated learning client who completes
     learning process containing two stages."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # Six is reasonable in this case.
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None,
                 personalized_model=None,
                 contrastive_transform=None):
        super().__init__(model, datasource, algorithm, trainer)

        # the specific transform used by the contrastive learning
        self.custom_contrastive_transform = contrastive_transform
        self.contrastive_transform = None

        # the unlabeledset in which samples do not have labels
        # in the self-supervised learning, there are many datasets,
        # such as STL10, containing samples without annotations. This
        # makes them contain two different datasets
        #   - trainset, samples with labels
        #   - unlabeled set, samples without labels
        # Therefore, apart from the trainset and testset, it is necessary
        # to load the unlabeled set when necessary (i.e., the corresponding
        # datasource contain the unlabeled set)
        self.unlabeledset = None
        self.unlabeled_sampler = None
        # using the name monitor_trainset is general in this domain,
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
        self.monitor_trainset = None
        self.eval_trainset = None

        # the personalized model here corresponds to the task-specific
        #   model in the general ssl
        self.custom_personalized_model = personalized_model
        self.personalized_model = None

    def configure(self) -> None:
        """ Performing the general client's configure and then initialize the
            personalized model for the client. """
        super().configure()
        if self.custom_personalized_model is not None:
            self.personalized_model = self.custom_model
            self.custom_personalized_model = None

        if self.personalized_model is None:

            encode_dim = self.trainer.model.encode_dim
            personalized_model_name = Config().trainer.personalized_model_name
            self.personalized_model = general_MLP_model.Model.get_model(
                model_type=personalized_model_name, input_dim=encode_dim)
            # present the personalzied model's info
            input_dim = self.personalized_model[0][0].in_features
            params = sum(p.numel()
                         for p in self.personalized_model.parameters()
                         if p.requires_grad)

            logging.info(
                "   [Client #%d]'s personalized model: Trainable Params[%s]",
                self.client_id, params)

        # assign the client's personalized model to its trainer
        self.trainer.set_client_personalized_model(self.personalized_model)

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        super().load_data()

        # obtain the contrastive trainset for SSL with data augmentation
        # transform. It is supported by the trainset
        # the characteristics:
        #   - SSL's data augmentation transform for contrastive training
        #   - trainset
        # use the custom contrastive transfrom is possible
        if self.custom_contrastive_transform is not None:
            self.contrastive_transform = self.custom_contrastive_transform
            self.custom_contrastive_transform = None

        # define the contrastive transform based on the requirement
        if self.contrastive_transform is None:
            augment_transformer_name = Config().data.augment_transformer_name
            self.contrastive_transform = get_aug(name=augment_transformer_name,
                                                 train=True,
                                                 for_downstream_task=False)

        # obtain the unlabeled set if it is supported by datasource
        if hasattr(self.datasource, 'get_unlabeled_set') and callable(
                self.datasource.get_unlabeled_set):

            self.unlabeledset = self.datasource.get_unlabeled_set()
            # Setting up the data sampler for the self.unlabeledset
            self.unlabeled_sampler = samplers_registry.get(
                self.datasource, self.client_id, testing="unlabelled")
            self.unlabeledset = datawrapper_registry.get(
                self.unlabeledset, self.contrastive_transform)
            logging.info(
                "[Client #%d] loaded the [%d] unlabeled dataset",
                self.client_id,
                int(len(self.unlabeledset) / Config().clients.total_clients))

        self.trainset = datawrapper_registry.get(self.trainset,
                                                 self.contrastive_transform)

        # get the same trainset again for monitor trainset
        # this dataset is prepared to monitor the representation learning
        # the characteristics:
        #   - utilize the general data transform, which the same as the transform in
        #       downstream task' test phase,
        #       i.e., test transform of the 'datasources/test_aug.py'
        #   - trainset
        #   - the transform for the monitor, such as knn
        monitor_augment_transformer = get_aug(name="test",
                                              train=False,
                                              for_downstream_task=False)

        self.monitor_trainset = self.datasource.get_train_set()
        self.monitor_trainset = datawrapper_registry.get(
            self.monitor_trainset, monitor_augment_transformer)

        if Config().clients.do_test:

            # obtain the testset with the corresponding transform for
            #   - the test loader for monitor
            #   - the test loader for downstream tasks, such as the
            #   the personalized learning of each client.
            # the characteristics:
            #   - the general data transform for upper mentioned two
            #   blocks
            #   - testset
            augment_transformer = get_aug(name="test",
                                          train=False,
                                          for_downstream_task=False)
            self.testset = datawrapper_registry.get(self.testset,
                                                    augment_transformer)

            # obtain the trainset with the corresponding transform for
            #   - the train loader for downstream tasks, such as the
            #   the personalized learning of each client
            # the characteristics:
            #   - the general data transform for the downstream tasks,
            #   such as the image classification
            #   - trainset
            # Note: we utilize the 'eval' as this stage's prefix just
            #   to follow the commonly utilized name in self-supervised
            #   learning (ssl). They utilize the 'linear evaluation' because
            #   the performance on downstream tasks is regarded as the
            #   evaluation fro the representation learning of ssl.
            #   Therefore, to make it consistent, we call it eval_trainset

            augment_transformer = get_aug(name="test",
                                          train=True,
                                          for_downstream_task=True)
            self.eval_trainset = self.datasource.get_train_set()
            self.eval_trainset = datawrapper_registry.get(
                self.eval_trainset, augment_transformer)

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info(
            fonts.colourize(
                f"[{self}] Started training in communication round #{self.current_round}."
            ))

        # Perform model training
        try:
            training_time = self.trainer.train(
                self.trainset,
                self.sampler,
                unlabeled_trainset=self.unlabeledset,
                unlabeled_sampler=self.unlabeled_sampler)
        except ValueError:
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        def test_logging(logging_str, accuracy):
            if hasattr(Config().trainer, 'target_perplexity'):
                logging.info("[%s] {%s}: %.2f", self, logging_str, accuracy)
            else:
                logging.info("[%s] {%s}: %.2f%%", self, logging_str,
                             100 * accuracy)

        # set the accuracy to be 0
        # even thouth we set the do_test to be True, the test procedure will not
        # be performed in every round. We just set it to be 0
        # if the test procedure is performes, the accuracy will be changed.
        accuracy = 0
        # Generate a report for the server, performing model testing if applicable
        if hasattr(Config().clients, 'do_test') and Config().clients.do_test:
            # This is the monitor test performed to measure the representation's quality
            # based on the cluster method, such as k-nearest neighbors (KNN). The detailed
            # procedures of this test are:
            # 1.- Extract representation from the monitor trainset based on the trained
            # encoder of the self-supervised methods. The monitor trainset is a duplicate
            # of the trainset but without applying the contrastive data augmentation.
            # It only utilizes the normal test transform, as shown in
            # 'datasources/augmentations/test_aug.py' .
            # 2.- Train the KNN method based on the extracted representation
            # of the monitor trainset.
            # 3.- Using the trained KNN method to classify the testset to obtain accuracy.

            if hasattr(
                    Config().clients, 'test_interval'
            ) and self.current_round % Config().clients.test_interval == 0:

                accuracy = self.trainer.test(
                    testset=self.testset,
                    sampler=self.testset_sampler,
                    monitor_trainset=self.monitor_trainset,
                    monitor_trainset_sampler=self.sampler,
                    current_round=self.current_round)
                logging_str = "Monitor test"
                test_logging(logging_str, accuracy)

            # In general, the performance of self-supervised learning methods is
            # measured by applying their encoder to extract representation for the
            # downstream tasks; thus, the quantity metrics are reported based on the
            # objective of these tasks. This is commonly called linear evaluation and
            # is conducted after completing the training of self-supervised learning
            # methods. However, in federated learning, it is expected to track its
            # performance after several rounds of communication. Therefore, in every
            # #eval_test_interval round, the linear evaluation will be conducted to
            # obtain accuracy or other metrics. The procedures are:
            # 1.- Design a simple personalized model (i.e., self.personalized_model)
            # for the client's downstream task, such as the image classification.
            # The classifier can be a one layer fully-connected layer.
            # 2.- Use the encoder of the trained self-supervised method as the
            # backbone to extract representation from the train data.
            # 3.- Combine the backbone and the designed model to perform the
            # downstream task. Thus, the input samples are processed by the
            # backbone to generate representation, which is used as input for
            # the designed model to complete the task. This encoder/backbone is
            # frozen without any changes. Only the designed model (self.personalized_model)
            # is optimized.
            if hasattr(Config().clients,
                       'eval_test_interval') and self.current_round % Config(
                       ).clients.eval_test_interval == 0:
                # it is important to also point out
                # which current is the personalized model
                # trained.
                accuracy = self.trainer.eval_test(
                    testset=self.testset,
                    sampler=self.testset_sampler,
                    eval_trainset=self.eval_trainset,
                    eval_trainset_sampler=self.sampler,
                    current_round=self.current_round)

                logging_str = "Evaluation test"
                test_logging(logging_str, accuracy)
            if accuracy == -1:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()
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
