"""
References:

Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models,"
in IWQoS 2021.

Shokri et al., "Membership Inference Attacks Against Machine Learning Models," in IEEE S&P 2017.

https://ieeexplore.ieee.org/document/9521274
https://arxiv.org/pdf/1610.05820.pdf
"""

import copy

import torch
from torch.utils.data import SubsetRandomSampler

from plato.config import Config
from plato.servers import fedavg

from .mia import launch_attack, train_attack_model


class Server(fedavg.Server):
    """A federated unlearning server of federated unlearning with local PGA."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # A dictionary that maps client IDs to their sample indices
        self.sample_indices = {}

    def weights_aggregated(self, updates):
        """Extract required information from client reports after aggregating weights."""
        for update in updates:
            if update.client_id not in self.sample_indices:
                self.sample_indices[update.client_id] = {}
            self.sample_indices[update.client_id]["learned_indices"] = (
                update.report.indices
            )
            self.sample_indices[update.client_id]["unlearned_indices"] = (
                update.report.deleted_indices
            )

    def _perform_mia(self):
        """Train and test attack model."""
        batch_size = Config().trainer.batch_size
        trainset = self.datasource.get_train_set()

        learned_indices = []
        unlearned_indices = []
        for c in self.sample_indices.values():
            learned_indices += list(set(c["learned_indices"]) - set(learned_indices))
            unlearned_indices += list(
                set(c["unlearned_indices"]) - set(unlearned_indices)
            )

        gen = torch.Generator()
        gen.manual_seed(Config().data.random_seed)
        learned_sampler = SubsetRandomSampler(learned_indices, generator=gen)
        unlearned_sampler = SubsetRandomSampler(unlearned_indices, generator=gen)

        # Member data, i.e., data seen so far by clients
        learned_dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=False,
            sampler=learned_sampler,
        )

        # Data for evaluation, i.e., data deleted by the unlearning client
        unlearned_dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=False,
            sampler=unlearned_sampler,
        )

        # Non-member data, i.e., test dataset
        out_dataset = self.datasource.get_test_set()
        out_dataloader = torch.utils.data.DataLoader(
            out_dataset, batch_size=batch_size, shuffle=False
        )

        shadow_model = self.get_shadow_model()
        target_model = self.get_target_model()

        attack_model = train_attack_model(
            shadow_model, learned_dataloader, out_dataloader
        )

        launch_attack(target_model, attack_model, unlearned_dataloader, out_dataloader)

    def get_shadow_model(self):
        """Load the shadow model, which is the current global model in this case."""
        return copy.deepcopy(self.trainer.model)

    def get_target_model(self):
        """Load the target model, which is the global model after unlearning."""
        return copy.deepcopy(self.trainer.model)
