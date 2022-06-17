"""
An honest-but-curious federated learning server which can
analyze periodic gradients from certain clients to
perform the gradient leakage attacks and
reconstruct the training data of the victim clients.


References:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.

https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?"
in Advances in Neural Information Processing Systems 2020.

https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf
"""
import asyncio
import logging
import math
import os
import sys
from evaluations import get_evaluation_dict
from collections import OrderedDict
from copy import deepcopy
from statistics import mean

import lpips
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from plato.config import Config
from plato.servers import fedavg
from plato.utils import csv_processor
from torchvision import transforms

from defense.GradDefense.compensate import denoise
from utils.modules import PatchedModule
from utils.utils import cross_entropy_for_onehot
from utils.utils import total_variation as TV

cross_entropy = torch.nn.CrossEntropyLoss()
tt = transforms.ToPILImage()
torch.manual_seed(Config().algorithm.random_seed)

partition_size = Config().data.partition_size
epochs = Config().trainer.epochs
batch_size = Config().trainer.batch_size
num_iters = Config().algorithm.num_iters
log_interval = Config().algorithm.log_interval
dlg_result_path = f"{Config().params['result_path']}"
dlg_result_file = f"{dlg_result_path}/{os.getpid()}_evals.csv"
dlg_result_headers = [
    "Iteration",
    "Loss",
    "Average MSE",
    "Average LPIPS",
    "Average PSNR (dB)",
    "Average SSIM",
    "Average Library SSIM",
]


class Server(fedavg.Server):
    """ An honest-but-curious federated learning server with gradient leakage attack. """

    def __init__(self, model, trainer):
        super().__init__(model=model, trainer=trainer)
        self.attack_method = 'DLG'
        if hasattr(Config().algorithm, 'attack_method'):
            if Config().algorithm.attack_method in ['DLG', 'iDLG', 'csDLG']:
                self.attack_method = Config().algorithm.attack_method
            else:
                sys.exit('Error: Unknown attack method.')
        self.share_gradients = True
        if hasattr(Config().algorithm, 'share_gradients') and not Config().algorithm.share_gradients:
            self.share_gradients = False
        self.match_weights = False
        if hasattr(Config().algorithm, 'match_weights') and Config().algorithm.match_weights:
            self.match_weights = True
        self.use_updates = True
        if hasattr(Config().algorithm, 'use_updates') and not Config().algorithm.use_updates:
            self.use_updates = False
        self.defense_method = 'no'
        if hasattr(Config().algorithm, 'defense'):
            self.defense_method = Config().algorithm.defense

    async def process_reports(self):
        """ Process the client reports: before aggregating their weights,
            perform the gradient leakage attacks and reconstruct the training data.
        """
        if self.current_round == Config().algorithm.attack_round:
            self.deep_leakage_from_gradients(self.updates)
        await self.aggregate_weights(self.updates)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging with optional compensation."""
        deltas_received = self.compute_weight_deltas(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        _scale = 0
        for i, update in enumerate(deltas_received):
            __, report, __, __ = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            if self.defense_method == 'GradDefense':
                _scale += len(deltas_received) * Config().algorithm.perturb_slices_num / Config().algorithm.slices_num \
                    * (Config().algorithm.scale ** 2) * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        if self.defense_method == 'GradDefense':
            update_perturbed = []
            for name, delta in avg_update.items():
                update_perturbed.append(delta)
            update_compensated = denoise(gradients=update_perturbed,
                                         scale=math.sqrt(_scale), Q=Config().algorithm.Q)
            for i, name in enumerate(avg_update.keys()):
                avg_update[name] = update_compensated[i]

        return avg_update

    def compute_weight_deltas(self, updates):
        """ Extract the model weight updates from client updates. """
        weights_received = [payload[0] for (__, __, payload, __) in updates]
        return self.algorithm.compute_weight_deltas(weights_received)

    def deep_leakage_from_gradients(self, updates):
        """ Analyze periodic gradients from certain clients. """
        # Process data from the victim client
        deltas_received = self.compute_weight_deltas(updates)
        __, __, payload, __ = updates[Config().algorithm.victim_client]
        # The ground truth should be used only for evaluation
        gt_data, gt_label, target_grad = payload[1]
        target_weights = payload[0]
        if not self.share_gradients and self.match_weights and self.use_updates:
            target_weights = deltas_received[Config().algorithm.victim_client]

        # Initialize the csv file
        csv_processor.initialize_csv(
            dlg_result_file, dlg_result_headers, dlg_result_path)

        # Assume the reconstructed data shape is known, which can be also derived from the target dataset
        num_images = partition_size
        data_size = [num_images, gt_data.shape[1],
                     gt_data.shape[2], gt_data.shape[3]]
        self.plot_gt(num_images, gt_data, gt_label)

        if not self.share_gradients and not self.match_weights:
            # Obtain the local updates from clients
            target_grad = []
            for delta in deltas_received[
                    Config().algorithm.victim_client].values():
                target_grad.append(-delta / Config().trainer.learning_rate)

            total_local_steps = epochs * math.ceil(partition_size / batch_size)
            target_grad = [x / total_local_steps for x in target_grad]

        # Generate dummy items and initialize optimizer
        dummy_data = torch.randn(data_size).to(
            Config().device()).requires_grad_(True)

        dummy_label = torch.randn(
            (num_images, Config().trainer.num_classes)).to(
                Config().device()).requires_grad_(True)

        if self.attack_method == 'DLG':
            match_optimizer = torch.optim.LBFGS(
                [dummy_data, dummy_label], lr=Config().algorithm.lr)
            est_label = [None] * num_images
            for i in range(num_images):
                logging.info("[%s Gradient Leakage Attacking with %s defense...] Dummy label is %d.",
                             self.attack_method, self.defense_method, torch.argmax(dummy_label[i], dim=-1).item())
        elif self.attack_method == 'iDLG':
            match_optimizer = torch.optim.LBFGS(
                [dummy_data, ], lr=Config().algorithm.lr)
            # Estimate the gt label
            est_label = torch.argmin(torch.sum(
                target_grad[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            for i in range(num_images):
                logging.info("[%s Gradient Leakage Attacking with %s defense...] Estimated label is %d.",
                             self.attack_method, self.defense_method, est_label.item())

        history, losses, mses, lpipss, psnrs, ssims, library_ssims = [], [], [], [], [], [], []
        avg_mses, avg_lpips, avg_psnr, avg_ssim, avg_library_ssim = [], [], [], [], []

        # Conduct gradients/weights/updates matching
        if not self.share_gradients and self.match_weights:
            model = deepcopy(self.trainer.model)
            closure = self.weight_closure(match_optimizer, dummy_data,
                                          dummy_label, target_weights, model)
        else:
            closure = self.gradient_closure(match_optimizer, dummy_data,
                                            dummy_label, est_label, target_grad)

        for iters in range(num_iters):
            match_optimizer.step(closure)
            current_loss = closure().item()
            losses.append(current_loss)

            if iters % log_interval == 0:
                # Finding evaluation metrics
                eval_dict = get_evaluation_dict(
                    dummy_data, gt_data, num_images)
                mses.append(eval_dict["mses"])
                lpipss.append(eval_dict["lpipss"])
                psnrs.append(eval_dict["psnrs"])
                ssims.append(eval_dict["ssims"])
                library_ssims.append(eval_dict["library_ssims"])
                avg_mses.append(eval_dict["avg_mses"])
                avg_lpips.append(eval_dict["avg_lpips"])
                avg_psnr.append(eval_dict["avg_psnr"])
                avg_ssim.append(eval_dict["avg_ssim"])
                avg_library_ssim.append(eval_dict["avg_library_ssim"])

                logging.info(
                    "[%s Gradient Leakage Attacking with %s defense...] Iter %d: Loss = %.10f, avg MSE = %.8f, avg LPIPS = %.8f, avg PSNR = %.4f dB, avg SSIM = %.3f, avg library SSIM = %.3f",
                    self.attack_method, self.defense_method, iters, losses[-1], avg_mses[-1], avg_lpips[-1], avg_psnr[-1], avg_ssim[-1], avg_library_ssim[-1])

                if self.attack_method == 'DLG':
                    history.append([[
                        tt(dummy_data[i][0].cpu()
                           ), torch.argmax(dummy_label[i], dim=-1).item(), dummy_data[i]
                    ] for i in range(num_images)])
                elif self.attack_method == 'iDLG':
                    history.append([[
                        tt(dummy_data[i][0].cpu()
                           ), est_label[i].item(), dummy_data[i]
                    ] for i in range(num_images)])

                new_row = [
                    iters,
                    round(losses[-1], 8),
                    round(avg_mses[-1], 8),
                    round(avg_lpips[-1], 8),
                    round(avg_psnr[-1], 4),
                    round(avg_ssim[-1], 3),
                    round(avg_library_ssim[-1], 3)
                ]
                csv_processor.write_csv(dlg_result_file, new_row)

        self.plot_reconstructed(num_images, history)

        # Save the tensors into a .pt file
        tensor_file_path = f"{dlg_result_path}/{os.getpid()}_tensors.pt"
        result = {
            i * log_interval: {j: history[i][j][0]
                               for j in range(num_images)}
            for i in range(len(history))
        }
        torch.save(result, tensor_file_path)

        logging.info("Attack complete")

    def gradient_closure(self, match_optimizer, dummy_data, dummy_label, est_label,
                         target_grad):
        """ Take a step to match the gradients. """

        def closure():
            match_optimizer.zero_grad()
            dummy_pred = self.trainer.model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            if self.attack_method == 'DLG':
                dummy_loss = cross_entropy_for_onehot(dummy_pred,
                                                      dummy_onehot_label)
            elif self.attack_method == 'iDLG':
                dummy_loss = cross_entropy(dummy_pred, est_label)

            dummy_grad = torch.autograd.grad(dummy_loss,
                                             self.trainer.model.parameters(),
                                             create_graph=True)

            rec_loss = self.reconstruction_costs([dummy_grad], target_grad)
            if hasattr(Config().algorithm, 'total_variation') and Config().algorithm.total_variation > 0:
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            return rec_loss

        return closure

    def weight_closure(self, match_optimizer, dummy_data, dummy_label,
                       target_weights, model):
        """ Take a step to match the weights. """

        def closure():
            match_optimizer.zero_grad()
            dummy_weight = self.loss_steps(dummy_data, dummy_label, model)

            rec_loss = self.reconstruction_costs(
                [dummy_weight], list(target_weights.values()))
            if hasattr(Config().algorithm, 'total_variation') and Config().algorithm.total_variation > 0:
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            return rec_loss

        return closure

    def loss_steps(self, dummy_data, dummy_label, model):
        """ Take a few gradient descent steps to fit the model to the given input. """
        patched_model = PatchedModule(model)
        if self.use_updates:
            patched_model_origin = deepcopy(patched_model)

        # TODO: optional parameters: lr_schedule, create_graph...

        # TODO: another parameter local steps instead of epoch here
        for epoch in range(epochs):
            if batch_size == 1:
                dummy_pred = patched_model(
                    dummy_data, patched_model.parameters)
                labels_ = dummy_label
            else:
                idx = epoch % (dummy_data.shape[0] // batch_size)
                dummy_pred = patched_model(dummy_data[idx * batch_size:(idx + 1) *
                                                      batch_size], patched_model.parameters)
                labels_ = dummy_label[idx * batch_size:(idx + 1) * batch_size]

            loss = cross_entropy(dummy_pred, torch.argmax(labels_,
                                                          dim=-1)).sum()

            grad = torch.autograd.grad(loss,
                                       patched_model.parameters.values(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)

            patched_model.parameters = OrderedDict(
                (name, param - Config().trainer.learning_rate * grad_part)
                for ((name, param),
                     grad_part) in zip(patched_model.parameters.items(), grad))
        if self.use_updates:
            patched_model.parameters = OrderedDict((name, param - param_origin)
                                                   for ((name, param), (name_origin, param_origin))
                                                   in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
        return list(patched_model.parameters.values())

    @staticmethod
    def reconstruction_costs(dummy, target):
        indices = torch.arange(len(target))
        cost_fn = Config().algorithm.cost_fn

        total_costs = 0
        for trial in dummy:
            pnorm = [0, 0]
            costs = 0
            for i in indices:
                if cost_fn == 'l2':
                    costs += ((trial[i] - target[i]).pow(2)).sum()
                elif cost_fn == 'l1':
                    costs += ((trial[i] - target[i]).abs()).sum()
                elif cost_fn == 'max':
                    costs += ((trial[i] - target[i]).abs()).max()
                elif cost_fn == 'sim':
                    costs -= (trial[i] * target[i]).sum()
                    pnorm[0] += trial[i].pow(2).sum()
                    pnorm[1] += target[i].pow(2).sum()
                elif cost_fn == 'simlocal':
                    costs += 1 - torch.nn.functional.cosine_similarity(
                        trial[i].flatten(), target[i].flatten(), 0, 1e-10)
            if cost_fn == 'sim':
                costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

            # Accumulate final costs
            total_costs += costs

        return total_costs / len(dummy)

    @staticmethod
    def plot_gt(num_images, gt_data, gt_label):
        """ Plot ground truth data. """
        gt_result_path = f"{Config().params['result_path']}/{os.getpid()}_gt.png"
        gt_figure = plt.figure(figsize=(12, 4))

        for i in range(num_images):
            current_label = torch.argmax(gt_label[i], dim=-1).item()
            logging.info("Ground truth labels: %d", current_label)
            gt_figure.add_subplot(1, num_images, i + 1)
            plt.imshow(tt(gt_data[i][0].cpu()))
            plt.axis('off')
            plt.title("GT image %d\nLabel: %d" % ((i + 1), current_label))
        plt.savefig(gt_result_path)

    @staticmethod
    def plot_reconstructed(num_images, history):
        """ Plot the reconstructed data. """
        reconstructed_result_path = f"{Config().params['result_path']}/{os.getpid()}_reconstructed.png"
        for i in range(num_images):
            logging.info("Reconstructed label is %d.", history[-1][i][1])

        fig = plt.figure(figsize=(12, 8))
        rows = math.ceil(len(history) / 2)
        outer = gridspec.GridSpec(rows, 2, wspace=0.2, hspace=0.2)

        for i in range(num_iters // log_interval):
            inner = gridspec.GridSpecFromSubplotSpec(1,
                                                     num_images,
                                                     subplot_spec=outer[i])
            outerplot = plt.Subplot(fig, outer[i])
            outerplot.set_title("Iter=%d" %
                                (i * log_interval))
            outerplot.axis('off')
            fig.add_subplot(outerplot)

            for j in range(num_images):
                innerplot = plt.Subplot(fig, inner[j])
                innerplot.imshow(history[i][j][0])
                innerplot.axis('off')
                fig.add_subplot(innerplot)
        fig.savefig(reconstructed_result_path)
