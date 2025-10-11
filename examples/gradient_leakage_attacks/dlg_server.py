"""
A federated learning server which can analyze periodic gradients
from certain clients to perform the gradient leakage attacks and
reconstruct the training data of the victim clients.

References:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.
https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?"
in Advances in Neural Information Processing Systems 2020.
https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf

Wen et al., "Fishing for User Data in Large-Batch Federated Learning via Gradient Magnification",
in Proceedings of the 39th International Conference on Machine Learning (ICML), 2022.
https://proceedings.mlr.press/v162/wen22a/wen22a.pdf
"""

import asyncio
import logging
import math
import os
import shutil
from collections import OrderedDict
from copy import deepcopy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from defense.GradDefense.compensate import denoise
from malicious_attacks.fishing import (
    check_with_tolerance,
    reconfigure_for_class_attack,
    reconfigure_for_feature_attack,
    reconstruct_feature,
)
from torchvision import transforms
from utils import consts
from utils.evaluations import get_evaluation_dict
from utils.helpers import cross_entropy_for_onehot
from utils.helpers import total_variation as TV
from utils.modules import PatchedModule

from plato.config import Config
from plato.servers import fedavg
from plato.utils import csv_processor

cross_entropy = torch.nn.CrossEntropyLoss(reduce="mean")
tt = transforms.ToPILImage()

partition_size = Config().data.partition_size
epochs = Config().trainer.epochs
batch_size = Config().trainer.batch_size
num_iters = Config().algorithm.num_iters
log_interval = Config().algorithm.log_interval
trials = Config().algorithm.trials
dlg_result_path = f"{Config().params['result_path']}/{os.getpid()}"
dlg_result_headers = [
    "Iteration",
    "Loss",
    "Average Data MSE",
    "Average Feature MSE",
    "Average LPIPS",
    "Average PSNR (dB)",
    "Average SSIM",
]


class Server(fedavg.Server):
    """An honest-but-curious federated learning server with gradient leakage attack."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.attack_method = None
        self.share_gradients = True
        if (
            hasattr(Config().algorithm, "share_gradients")
            and not Config().algorithm.share_gradients
        ):
            self.share_gradients = False
        self.match_weights = False
        if (
            hasattr(Config().algorithm, "match_weights")
            and Config().algorithm.match_weights
        ):
            self.match_weights = True
        self.use_updates = True
        if (
            hasattr(Config().algorithm, "use_updates")
            and not Config().algorithm.use_updates
        ):
            self.use_updates = False
        self.defense_method = "no"
        if hasattr(Config().algorithm, "defense"):
            if Config().algorithm.defense in [
                "GradDefense",
                "Soteria",
                "GC",
                "DP",
                "Outpost",
            ]:
                self.defense_method = Config().algorithm.defense
            else:
                logging.info("No Defense Applied")
        self.best_mse = math.inf
        # Save trail 1 as the best as default when results are all bad
        self.best_trial = 1
        self.iter = 0
        self.labels_ = None

        self.gt_data = None
        self.gt_labels = None
        self.target_grad = None
        self.target_weights = None
        # Assume the reconstructed data shape is known, which can be also derived from the target dataset
        self.num_images = partition_size
        self.dm = None
        self.ds = None

        # Fishing attack related
        self.target_cls = 0
        self.target_indx = None
        self.single_gradient_recovered = False
        self.feature_within_tolerance = False
        self.all_feature_val = []
        self.feature_loc = None
        self.start_round = None
        self.rec_round = 0
        self.modified_model_states = None

    def weights_received(self, weights_received):
        """
        Perform attack in attack around after the updated weights have been aggregated.
        """
        weights_received = [payload[0] for payload in weights_received]
        if Config().algorithm.attack_method in ["DLG", "iDLG", "csDLG"]:
            self.attack_method = Config().algorithm.attack_method
            self._deep_leakage_from_gradients(weights_received)

        return weights_received

    async def aggregate_deltas(self, updates, deltas_received):
        """
        Aggregate weight updates from the clients using
        federated averaging with optional compensation.
        """
        # Extract the total number of samples
        self.total_samples = sum([update.report.num_samples for update in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        _scale = 0
        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            if self.defense_method == "GradDefense":
                _scale += (
                    len(deltas_received)
                    * Config().algorithm.perturb_slices_num
                    / Config().algorithm.slices_num
                    * (Config().algorithm.scale ** 2)
                    * (num_samples / self.total_samples)
                )

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        if self.defense_method == "GradDefense":
            update_perturbed = []
            for name, delta in avg_update.items():
                update_perturbed.append(delta)
            update_compensated = denoise(
                gradients=update_perturbed,
                scale=math.sqrt(_scale),
                q=Config().algorithm.Q,
            )
            for i, name in enumerate(avg_update.keys()):
                avg_update[name] = update_compensated[i]

        return avg_update

    def customize_server_payload(self, payload):
        """Customizes the server payload before sending to the client."""
        if self.start_round is None:
            self.start_round = self.current_round
        if (
            hasattr(Config().algorithm, "fishing")
            and Config().algorithm.fishing
            and self.current_round > self.start_round
        ):
            self.algorithm.load_weights(self.modified_model_states)
            payload = self.algorithm.extract_weights()
        return payload

    def _deep_leakage_from_gradients(self, weights_received):
        """Analyze periodic gradients from certain clients."""
        # Process data from the victim client
        # The ground truth should be used only for evaluation
        baseline_weights = self.algorithm.extract_weights()
        deltas_received = self.algorithm.compute_weight_deltas(
            baseline_weights, weights_received
        )
        update = self.updates[Config().algorithm.victim_client]

        if self.current_round == self.start_round:
            self.gt_data, self.gt_labels = (
                update.payload[1].to(Config().device()),
                update.payload[2].to(Config().device()),
            )
            # Mean and std of data
            if Config().data.datasource == "CIFAR10":
                data_mean = consts.cifar10_mean
                data_std = consts.cifar10_std
            elif Config().data.datasource == "CIFAR100":
                data_mean = consts.cifar100_mean
                data_std = consts.cifar100_std
            elif Config().data.datasource == "TinyImageNet":
                data_mean = consts.imagenet_mean
                data_std = consts.imagenet_std
            elif Config().data.datasource == "MNIST":
                data_mean = consts.mnist_mean
                data_std = consts.mnist_std
            elif Config().data.datasource == "EMNIST":
                data_mean = consts.emnist_mean
                data_std = consts.emnist_std
            self.dm = torch.as_tensor(
                data_mean, device=Config().device(), dtype=torch.float
            )[:, None, None]
            self.ds = torch.as_tensor(
                data_std, device=Config().device(), dtype=torch.float
            )[:, None, None]

            gt_result_path = f"{dlg_result_path}/ground_truth.pdf"
            self._make_plot(
                self.num_images,
                self.gt_data,
                gt_result_path,
                self.dm,
                self.ds,
            )

        # Obtain target weight updates if matching updates
        self.target_weights = None
        if not self.share_gradients and self.match_weights:
            if self.use_updates:
                self.target_weights = deltas_received[Config().algorithm.victim_client]
            else:
                self.target_weights = update.payload[0]
            # ignore running statistics in state_dict()
            states_to_save = []
            for name, _ in self.trainer.model.named_parameters():
                states_to_save.append(name)
            states_to_remove = []
            for name in self.target_weights.keys():
                if name not in states_to_save:
                    states_to_remove.append(name)
            for name in states_to_remove:
                del self.target_weights[name]
            self.target_weights = [
                weight.to(Config().device()) for weight in self.target_weights.values()
            ]

        # Obtain target gradients if matching gradients
        self.target_grad = [grad.to(Config().device()) for grad in update.payload[3]]
        if not self.share_gradients and not self.match_weights:
            # Obtain the local updates from clients
            self.target_grad = []
            for delta in deltas_received[Config().algorithm.victim_client].values():
                self.target_grad.append(
                    -delta.to(Config().device()) / Config().parameters.optimizer.lr
                )

            total_local_steps = epochs * math.ceil(partition_size / batch_size)
            self.target_grad = [x / total_local_steps for x in self.target_grad]

        if hasattr(Config().algorithm, "fishing") and Config().algorithm.fishing:
            self.fishing_attack(self.target_grad, self.target_weights, self.gt_labels)

        # Optimization-based attack
        if (
            not (hasattr(Config().algorithm, "fishing") and Config().algorithm.fishing)
            or self.current_round == self.rec_round
        ):
            # Generate dummy items and initialize optimizer
            torch.manual_seed(Config().algorithm.random_seed)

            logging.info("Running %d Trials", trials)
            for trial_number in range(trials):
                self.run_trial(
                    trial_number,
                    self.num_images,
                    self.target_weights,
                    self.target_grad,
                    self.gt_data,
                    self.gt_labels,
                    self.dm,
                    self.ds,
                )

            self._save_best()

            logging.info("Attack finished.")
            self.current_round = Config().trainer.rounds + 1

    def run_trial(
        self,
        trial_number,
        num_images,
        target_weights,
        target_grad,
        gt_data,
        gt_labels,
        dm,
        ds,
    ):
        """Run the attack for one trial."""
        logging.info("Starting Attack Number %d", (trial_number + 1))

        trial_result_path = f"{dlg_result_path}/t{trial_number + 1}"
        trial_csv_file = f"{trial_result_path}/evals.csv"

        # Initialize the csv file
        csv_processor.initialize_csv(
            trial_csv_file, dlg_result_headers, trial_result_path
        )

        if Config().algorithm.init_data == "randn":
            dummy_data = (
                torch.randn(
                    [num_images, gt_data.shape[1], gt_data.shape[2], gt_data.shape[3]]
                )
                .to(Config().device())
                .requires_grad_(True)
            )
        elif Config().algorithm.init_data == "rand":
            dummy_data = (
                (
                    (
                        torch.rand(
                            [
                                num_images,
                                gt_data.shape[1],
                                gt_data.shape[2],
                                gt_data.shape[3],
                            ]
                        )
                        - 0.5
                    )
                    * 2
                )
                .to(Config().device())
                .requires_grad_(True)
            )
        elif Config().algorithm.init_data == "zeros":
            dummy_data = (
                torch.zeros(
                    [num_images, gt_data.shape[1], gt_data.shape[2], gt_data.shape[3]]
                )
                .to(Config().device())
                .requires_grad_(True)
            )
        elif Config().algorithm.init_data == "half":
            dummy_data = (
                (
                    torch.ones(
                        [
                            num_images,
                            gt_data.shape[1],
                            gt_data.shape[2],
                            gt_data.shape[3],
                        ]
                    )
                    - 0.5
                )
                .to(Config().device())
                .requires_grad_(True)
            )

        dummy_labels = (
            torch.randn((num_images, Config().parameters.model.num_classes))
            .to(Config().device())
            .requires_grad_(True)
        )

        if self.attack_method == "DLG":
            param = [dummy_data, dummy_labels]
        elif self.attack_method in ["iDLG", "csDLG"]:
            param = [
                dummy_data,
            ]

        # Init reconstruction optimizer
        if Config().algorithm.rec_optim == "Adam":
            match_optimizer = torch.optim.Adam(param, lr=Config().algorithm.rec_lr)
        elif Config().algorithm.rec_optim == "SGD":
            match_optimizer = torch.optim.SGD(
                param, lr=0.01, momentum=0.9, nesterov=True
            )
        elif Config().algorithm.rec_optim == "LBFGS":
            match_optimizer = torch.optim.LBFGS(param, lr=Config().algorithm.rec_lr)

        # Init learning rate scheduler
        if Config().algorithm.lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                match_optimizer,
                milestones=[
                    num_iters // 2.667,
                    num_iters // 1.6,
                    num_iters // 1.142,
                ],
                gamma=0.1,
            )  # 3/8 5/8 7/8

        if self.attack_method == "DLG":
            labels_ = dummy_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Dummy label is %d.",
                    self.attack_method,
                    trial_number,
                    self.defense_method,
                    torch.argmax(dummy_labels[i], dim=-1).item(),
                )

        elif self.attack_method == "iDLG":
            # Estimate the gt label
            est_labels = (
                torch.argmin(torch.sum(target_grad[-2], dim=-1), dim=-1)
                .detach()
                .reshape((1,))
                .requires_grad_(False)
            )
            labels_ = est_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Estimated label is %d.",
                    self.attack_method,
                    trial_number,
                    self.defense_method,
                    est_labels.item(),
                )
        elif self.attack_method == "csDLG":
            labels_ = gt_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Known label is %d.",
                    self.attack_method,
                    trial_number,
                    self.defense_method,
                    torch.argmax(gt_labels[i], dim=-1).item(),
                )

        history, losses = [], []
        avg_data_mses, avg_feat_mses, avg_lpips, avg_psnr, avg_ssim = [], [], [], [], []

        # Conduct gradients/weights/updates matching
        if not self.share_gradients and self.match_weights:
            model = deepcopy(self.trainer.model.to(Config().device()))
            closure = self._weight_closure(
                match_optimizer, dummy_data, labels_, target_weights, model
            )
        else:
            closure = self._gradient_closure(
                match_optimizer, dummy_data, labels_, target_grad
            )

        early_exit = False
        for self.iter in range(num_iters):
            current_loss = match_optimizer.step(closure)
            losses.append(current_loss.item())

            if Config().algorithm.lr_decay:
                scheduler.step()

            # Project into image space
            with torch.no_grad():
                if Config().algorithm.boxed:
                    dummy_data.data = torch.max(
                        torch.min(dummy_data, (1 - dm) / ds), -dm / ds
                    )

                if math.isnan(current_loss):
                    logging.info("Not a number, ending this attack attempt")
                    early_exit = True
                    break

                if self.iter % log_interval == 0:
                    eval_dict = get_evaluation_dict(
                        dummy_data,
                        gt_data,
                        num_images,
                        self.trainer.model.to(Config().device()),
                        ds,
                    )
                    avg_data_mses.append(eval_dict["avg_data_mses"])
                    avg_feat_mses.append(eval_dict["avg_feat_mses"])
                    avg_lpips.append(eval_dict["avg_lpips"])
                    avg_psnr.append(eval_dict["avg_psnr"])
                    avg_ssim.append(eval_dict["avg_ssim"])

                    logging.info(
                        "[%s Gradient Leakage Attack %d with %s defense...] Iter %d: "
                        "Loss = %.4f, avg Data MSE = %.4f, avg Feature MSE = %.4f, "
                        "avg LPIPS = %.4f, avg PSNR = %.4f dB, avg SSIM = %.4f",
                        self.attack_method,
                        (trial_number + 1),
                        self.defense_method,
                        self.iter,
                        losses[-1],
                        avg_data_mses[-1],
                        avg_feat_mses[-1],
                        avg_lpips[-1],
                        avg_psnr[-1],
                        avg_ssim[-1],
                    )

                    if self.attack_method == "DLG":
                        history.append(
                            [
                                [
                                    dummy_data[i],
                                    torch.argmax(dummy_labels[i], dim=-1).item(),
                                ]
                                for i in range(num_images)
                            ]
                        )
                    elif self.attack_method == "iDLG":
                        history.append(
                            [
                                [
                                    dummy_data[i],
                                    est_labels[i].item(),
                                ]
                                for i in range(num_images)
                            ]
                        )
                    elif self.attack_method == "csDLG":
                        history.append(
                            [
                                [
                                    dummy_data[i],
                                    torch.argmax(gt_labels[i], dim=-1).item(),
                                ]
                                for i in range(num_images)
                            ]
                        )

                    new_row = [
                        self.iter,
                        round(losses[-1], 4),
                        round(avg_data_mses[-1], 4),
                        round(avg_feat_mses[-1], 4),
                        round(avg_lpips[-1], 4),
                        round(avg_psnr[-1], 4),
                        round(avg_ssim[-1], 3),
                    ]
                    csv_processor.write_csv(trial_csv_file, new_row)

        if not early_exit:
            with torch.no_grad():
                # TODO: use other scoring criteria
                if self.best_mse > avg_data_mses[-1]:
                    self.best_mse = avg_data_mses[-1]
                    self.best_trial = (
                        trial_number + 1
                    )  # the +1 is because we index from 1 and not 0

                reconstructed_path = (
                    f"{trial_result_path}/reconstruction_iterations.png"
                )
                self._plot_reconstructed(
                    num_images, history, reconstructed_path, dm, ds
                )
                final_tensor = torch.stack(
                    [history[-1][i][0] for i in range(num_images)]
                )
                final_result_path = f"{trial_result_path}/final_attack_result.pdf"
                self._make_plot(num_images, final_tensor, final_result_path, dm, ds)

                # Save the tensors into a .pt file
                tensor_file_path = f"{trial_result_path}/tensors.pt"
                result = {
                    i * log_interval: {
                        j: history[i][j][0].cpu().permute(1, 2, 0)
                        for j in range(num_images)
                    }
                    for i in range(len(history))
                }
                torch.save(result, tensor_file_path)

                logging.info("Attack %d complete", (trial_number + 1))

    def _gradient_closure(self, match_optimizer, dummy_data, labels, target_grad):
        """Take a step to match the gradients."""

        def closure():
            match_optimizer.zero_grad()
            self.trainer.model.to(Config().device())
            # Set model mode for dummy data optimization
            if (
                hasattr(Config().algorithm, "dummy_eval")
                and Config().algorithm.dummy_eval
            ):
                self.trainer.model.eval()
            else:
                self.trainer.model.train()
            self.trainer.model.zero_grad()

            dummy_pred = self.trainer.model(dummy_data)

            if self.attack_method == "DLG":
                dummy_onehot_label = F.softmax(labels, dim=-1)
                dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label)
            elif self.attack_method in "iDLG":
                dummy_loss = cross_entropy(dummy_pred, labels)
            elif self.attack_method in "csDLG":
                dummy_loss = cross_entropy(dummy_pred, torch.argmax(labels, dim=-1))

            dummy_grad = torch.autograd.grad(
                dummy_loss, self.trainer.model.parameters(), create_graph=True
            )

            rec_loss = self._reconstruction_costs([dummy_grad], target_grad)
            if (
                hasattr(Config().algorithm, "total_variation")
                and Config().algorithm.total_variation > 0
            ):
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            if self.attack_method == "csDLG":
                if (
                    hasattr(Config().algorithm, "signed")
                    and Config().algorithm.signed == "soft"
                ):
                    scaling_factor = 1 - self.iter / num_iters
                    dummy_data.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                else:
                    dummy_data.grad.sign_()
            return rec_loss

        return closure

    def _weight_closure(
        self, match_optimizer, dummy_data, labels, target_weights, model
    ):
        """Take a step to match the weights."""

        def closure():
            match_optimizer.zero_grad()

            # Set model mode for dummy data optimization
            if (
                hasattr(Config().algorithm, "dummy_eval")
                and Config().algorithm.dummy_eval
            ):
                model.eval()
            else:
                model.train()

            dummy_weight = self._loss_steps(dummy_data, labels, model)

            rec_loss = self._reconstruction_costs([dummy_weight], target_weights)
            if (
                hasattr(Config().algorithm, "total_variation")
                and Config().algorithm.total_variation > 0
            ):
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            return rec_loss

        return closure

    def _loss_steps(self, dummy_data, labels, model):
        """Take a few gradient descent steps to fit the model to the given input."""
        patched_model = PatchedModule(model)

        if self.use_updates:
            patched_model_origin = deepcopy(patched_model)

        for _ in range(epochs):
            for idx in range(int(math.ceil(dummy_data.shape[0] / batch_size))):
                dummy_pred = patched_model(
                    dummy_data[idx * batch_size : (idx + 1) * batch_size],
                    patched_model.parameters,
                )
                labels_ = labels[idx * batch_size : (idx + 1) * batch_size]

                loss = cross_entropy(dummy_pred, labels_).sum()

                grad = torch.autograd.grad(
                    loss,
                    patched_model.parameters.values(),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                )

                patched_model.parameters = OrderedDict(
                    (name, param - Config().parameters.optimizer.lr * grad_part)
                    for ((name, param), grad_part) in zip(
                        patched_model.parameters.items(), grad
                    )
                )
        if self.use_updates:
            patched_model.parameters = OrderedDict(
                (name, param - param_origin)
                for ((name, param), (name_origin, param_origin)) in zip(
                    patched_model.parameters.items(),
                    patched_model_origin.parameters.items(),
                )
            )
        return list(patched_model.parameters.values())

    def fishing_attack(self, target_grad, target_weights, gt_labels):
        """The fishing attack (https://github.com/JonasGeiping/breaching)."""
        if self.current_round == self.start_round:
            # Query the labels
            t_labels = torch.argmax(gt_labels, dim=-1).detach().cpu().numpy()
            logging.info(f"Found labels {t_labels} in first query.")

            # Find the target class index
            self.target_cls = Config().algorithm.target_cls
            self.target_indx = np.where(t_labels == self.target_cls)[0]
            if self.target_indx.size == 0:
                self.target_cls = np.unique(t_labels)[Config().algorithm.target_cls_idx]
                self.target_indx = np.where(t_labels == self.target_cls)[0]
            self.labels_ = torch.argmax(gt_labels, dim=-1)[self.target_indx]

            # Plot the images of the target class too for fishing attack
            if hasattr(Config().algorithm, "fishing") and Config().algorithm.fishing:
                for i in self.target_indx:
                    gt_target_cls = self.gt_data[i].unsqueeze(0)
                    gt_cls_path = f"{dlg_result_path}/gt_target_cls_{self.target_cls}_indx_{i}.pdf"
                    self._make_plot(
                        1,
                        gt_target_cls,
                        gt_cls_path,
                        self.dm,
                        self.ds,
                    )

        if self.current_round == self.start_round and len(self.target_indx) == 1:
            # simple cls attack if there is no cls collision
            logging.info("Attacking label %d with cls attack.", self.labels_.item())

            # modify the parameters first
            self.modified_model_states = reconfigure_for_class_attack(
                self.trainer.model, target_classes=self.target_cls
            )

            # Only target one data
            self.num_images = 1
            self.gt_labels = gt_labels[self.target_indx[0]].unsqueeze(0)
            self.rec_round = self.current_round + 1

        elif len(self.target_indx) > 1:
            # send several queries because of cls collision
            if self.current_round == self.start_round:
                logging.info(
                    "Attacking label %d with binary attack.", self.labels_[0].item()
                )
                num_collisions = (self.labels_ == self.target_cls).sum()
                logging.info(
                    f"There are in total {num_collisions.item()} datapoints with label {self.target_cls}."
                )

                # find the starting point and the feature entry gives the max avg value
                self.modified_model_states = reconfigure_for_class_attack(
                    self.trainer.model, target_classes=self.target_cls
                )
            else:
                # binary attack to recover all single gradients
                avg_feature = torch.flatten(
                    reconstruct_feature(
                        target_grad,
                        target_weights,
                        self.target_cls,
                    )
                )
                if self.feature_loc is None:
                    self.feature_loc = int(torch.argmax(avg_feature))
                feature_val = float(avg_feature[self.feature_loc])
                logging.info("Found avg feature val %.2f.", feature_val)

                if check_with_tolerance(
                    feature_val,
                    self.all_feature_val,
                    threshold=Config().algorithm.feat_threshold,
                ):
                    if not self.share_gradients and self.match_weights:
                        target = target_weights
                    else:
                        target = target_grad
                    curr_grad = list(target)
                    curr_grad[-1] = curr_grad[-1] * len(self.target_indx)
                    curr_grad[:-1] = [
                        grad_ii
                        * len(self.target_indx)
                        / Config().algorithm.feat_multiplier
                        for grad_ii in curr_grad[:-1]
                    ]
                    recovered_single_gradients = [curr_grad]
                    # return to the model with multiplier=1, (better with larger multiplier, but not optimizable if it is too large)
                    self.modified_model_states = reconfigure_for_feature_attack(
                        self.trainer.model,
                        feature_val,
                        self.feature_loc,
                        target_classes=self.target_cls,
                        allow_reset_param_weights=True,
                    )
                    self.algorithm.load_weights(self.modified_model_states)

                    # add reversed() because the ith is always more confident than i-1th
                    if not self.share_gradients and self.match_weights:
                        self.target_weights = list(
                            reversed(recovered_single_gradients)
                        )[Config().algorithm.grad_idx]
                    else:
                        self.target_grad = list(reversed(recovered_single_gradients))[
                            Config().algorithm.grad_idx
                        ]

                    # Only target one data
                    self.num_images = 1
                    self.gt_labels = gt_labels[self.target_indx[0]].unsqueeze(0)
                    self.rec_round = self.current_round
                else:
                    self.all_feature_val.append(feature_val)
                    logging.info(
                        "Querying feature %d with feature val %.2f.",
                        self.feature_loc,
                        feature_val,
                    )
                    self.modified_model_states = reconfigure_for_feature_attack(
                        self.trainer.model,
                        feature_val,
                        self.feature_loc,
                        target_classes=self.target_cls,
                    )

    def _save_best(self):
        src_folder = f"{dlg_result_path}/t{self.best_trial}"
        dst_folder = f"{dlg_result_path}/best(t{self.best_trial})"

        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        for file_name in os.listdir(src_folder):
            src = os.path.join(src_folder, file_name)
            dst = os.path.join(dst_folder, file_name)
            if os.path.isfile(src):
                shutil.copy(src, dst)

    @staticmethod
    def _reconstruction_costs(dummy, target):
        indices = torch.arange(len(target))

        ex = target[0]
        if Config().algorithm.cost_weights == "linear":
            weights = torch.arange(
                len(target), 0, -1, dtype=ex.dtype, device=ex.device
            ) / len(target)
        elif Config().algorithm.cost_weights == "exp":
            weights = torch.arange(len(target), 0, -1, dtype=ex.dtype, device=ex.device)
            weights = weights.softmax(dim=0)
            weights = weights / weights[0]
        else:
            weights = target[0].new_ones(len(target))

        cost_fn = Config().algorithm.cost_fn

        total_costs = 0
        for trial in dummy:
            pnorm = [0, 0]
            costs = 0
            for i in indices:
                if cost_fn == "l2":
                    costs += ((trial[i] - target[i]).pow(2)).sum() * weights[i]
                elif cost_fn == "l1":
                    costs += ((trial[i] - target[i]).abs()).sum() * weights[i]
                elif cost_fn == "max":
                    costs += ((trial[i] - target[i]).abs()).max() * weights[i]
                elif cost_fn == "sim":
                    costs -= (trial[i] * target[i]).sum() * weights[i]
                    pnorm[0] += trial[i].pow(2).sum() * weights[i]
                    pnorm[1] += target[i].pow(2).sum() * weights[i]
                elif cost_fn == "simlocal":
                    costs += (
                        1
                        - F.cosine_similarity(
                            trial[i].flatten(), target[i].flatten(), 0, 1e-10
                        )
                        * weights[i]
                    )
            if cost_fn == "sim":
                costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

            # Accumulate final costs
            total_costs += costs

        return total_costs / len(dummy)

    @staticmethod
    def _make_plot(num_images, image_data, path, dm, ds):
        """Plot image data."""

        if not os.path.exists(dlg_result_path):
            os.makedirs(dlg_result_path)

        rows, cols = None, None
        if hasattr(Config().results, "rows"):
            rows = Config().results.rows
        if hasattr(Config().results, "cols"):
            cols = Config().results.cols

        if rows is not None and cols is None:
            cols = math.ceil(num_images / rows)
        elif cols is not None and rows is None:
            rows = math.ceil(num_images / cols)
        elif rows is None and cols is None:
            # make the image wider by default
            # if you want the image to be taller by default then
            # switch the assignment statement for rows and cols variables
            logging.info("Using default dimensions for images")
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)
        elif (rows * cols) < num_images:
            logging.info("Row and column provided for plotting images is too small")
            logging.info("Using default dimensions for images")
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)

        scale_factor = rows + cols
        image_height = 16 * rows / scale_factor
        image_width = 16 * cols / scale_factor
        product = rows * cols

        image_data = image_data.detach().clone()
        image_data.mul_(ds).add_(dm).clamp_(0, 1)
        if num_images == 1:
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(image_data[0].permute(1, 2, 0).cpu())
            plt.axis("off")
        else:
            fig, axes = plt.subplots(
                nrows=rows,
                ncols=cols,
                gridspec_kw=dict(
                    wspace=0.0,
                    hspace=0.0,
                    top=1.0 - 0.5 / (rows + 1),
                    bottom=0.5 / (rows + 1),
                    left=0.5 / (cols + 1),
                    right=1 - 0.5 / (cols + 1),
                ),
                figsize=(image_width, image_height),
                sharey="row",
                sharex="col",
            )
            for i, img in enumerate(image_data):
                axes.ravel()[i].imshow(img.permute(1, 2, 0).cpu())
                axes.ravel()[i].set_axis_off()
            for i in range(num_images, product):
                axes.ravel()[i].set_axis_off()

        plt.tight_layout()
        plt.savefig(path)

    @staticmethod
    def _plot_reconstructed(num_images, history, reconstructed_result_path, dm, ds):
        """Plot the reconstructed data."""
        for i in range(num_images):
            logging.info("Reconstructed label is %d.", history[-1][i][1])

        fig = plt.figure(figsize=(12, 8))
        rows = math.ceil(len(history) / 2)
        outer = gridspec.GridSpec(rows, 2, wspace=0.2, hspace=0.2)

        for i, _ in enumerate(history):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, num_images, subplot_spec=outer[i]
            )
            outerplot = plt.Subplot(fig, outer[i])
            outerplot.set_title("Iter=%d" % (i * log_interval))
            outerplot.axis("off")
            fig.add_subplot(outerplot)

            for j in range(num_images):
                innerplot = plt.Subplot(fig, inner[j])
                innerplot.imshow(
                    history[i][j][0]
                    .detach()
                    .clone()
                    .mul_(ds)
                    .add_(dm)
                    .clamp_(0, 1)
                    .permute(1, 2, 0)
                    .cpu()
                )
                innerplot.axis("off")
                fig.add_subplot(innerplot)
        fig.savefig(reconstructed_result_path)
