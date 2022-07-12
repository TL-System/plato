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
import shutil
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
dlg_result_path = f"{Config().params['result_path']}/{os.getpid()}"
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
        if hasattr(
                Config().algorithm,
                'share_gradients') and not Config().algorithm.share_gradients:
            self.share_gradients = False
        self.match_weights = False
        if hasattr(Config().algorithm,
                   'match_weights') and Config().algorithm.match_weights:
            self.match_weights = True
        self.use_updates = True
        if hasattr(Config().algorithm,
                   'use_updates') and not Config().algorithm.use_updates:
            self.use_updates = False
        self.defense_method = 'no'
        if hasattr(Config().algorithm, 'defense'):
            if Config().algorithm.defense in ['GradDefense', 'Soteria', 'MC', 'DP']:
                self.defense_method = Config().algorithm.defense
            else:
                logging.info("No Defense Applied")
        self.best_mse = math.inf
        self.best_trial = 0

    async def process_reports(self):
        """ Process the client reports: before aggregating their weights,
            perform the gradient leakage attacks and reconstruct the training data.
        """
        if self.current_round == Config().algorithm.attack_round:
            self.deep_leakage_from_gradients(self.updates)
        await self.aggregate_weights(self.updates)

        # Testing the global model accuracy
        if hasattr(Config().server, 'do_test') and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info('[%s] Average client accuracy: %.2f%%.', self,
                         100 * self.accuracy)
        else:
            # Testing the updated model directly at the server
            self.accuracy = await self.trainer.server_test(
                self.testset, self.testset_sampler)

        if hasattr(Config().trainer, 'target_perplexity'):
            logging.info('[%s] Global model perplexity: %.2f\n', self,
                         self.accuracy)
        else:
            logging.info('[%s] Global model accuracy: %.2f%%\n', self,
                         100 * self.accuracy)

        await self.wrap_up_processing_reports()

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
                                         scale=math.sqrt(_scale),
                                         Q=Config().algorithm.Q)
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
        gt_data, gt_labels, target_grad = payload[1]
        target_weights = payload[0]
        if not self.share_gradients and self.match_weights and self.use_updates:
            target_weights = deltas_received[Config().algorithm.victim_client]

        # Assume the reconstructed data shape is known, which can be also derived from the target dataset
        num_images = partition_size
        data_size = [
            num_images, gt_data.shape[1], gt_data.shape[2], gt_data.shape[3]
        ]
        self.plot_gt(num_images, gt_data, gt_labels)

        # The number of restarts
        trials = 1
        if hasattr(Config().algorithm, "trials"):
            trials = Config().algorithm.trials

        logging.info("Running %d Trials", trials)

        if not self.share_gradients and not self.match_weights:
            # Obtain the local updates from clients
            target_grad = []
            for delta in deltas_received[
                    Config().algorithm.victim_client].values():
                target_grad.append(-delta / Config().trainer.learning_rate)

            total_local_steps = epochs * math.ceil(partition_size / batch_size)
            target_grad = [x / total_local_steps for x in target_grad]

        for trial_number in range(trials):
            self.run_trial(trial_number, num_images, data_size, target_weights,
                           target_grad, gt_data, gt_labels)

        self.save_best()

    def run_trial(self, trial_number, num_images, data_size, target_weights,
                  target_grad, gt_data, gt_labels):
        """ Run the attack for one trial. """
        logging.info("Starting Attack Number %d", (trial_number + 1))

        trial_result_path = f"{dlg_result_path}/t{trial_number + 1}"
        trial_csv_file = f"{trial_result_path}/evals.csv"

        # Initialize the csv file
        csv_processor.initialize_csv(trial_csv_file, dlg_result_headers,
                                     trial_result_path)

        # Generate dummy items and initialize optimizer
        dummy_data = torch.randn(data_size).to(
            Config().device()).requires_grad_(True)

        dummy_labels = torch.randn(
            (num_images, Config().trainer.num_classes)).to(
                Config().device()).requires_grad_(True)

        if self.attack_method == 'DLG':
            match_optimizer = torch.optim.LBFGS([dummy_data, dummy_labels],
                                                lr=Config().algorithm.lr)
            labels_ = dummy_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Dummy label is %d.",
                    self.attack_method, trial_number, self.defense_method,
                    torch.argmax(dummy_labels[i], dim=-1).item())

        elif self.attack_method == 'iDLG':
            match_optimizer = torch.optim.LBFGS([
                dummy_data,
            ],
                                                lr=Config().algorithm.lr)
            # Estimate the gt label
            est_labels = torch.argmin(torch.sum(target_grad[-2], dim=-1),
                                      dim=-1).detach().reshape(
                                          (1, )).requires_grad_(False)
            labels_ = est_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Estimated label is %d.",
                    self.attack_method, trial_number, self.defense_method,
                    est_labels.item())
        elif self.attack_method == 'csDLG':
            match_optimizer = torch.optim.LBFGS([
                dummy_data,
            ],
                                                lr=Config().algorithm.lr)
            labels_ = gt_labels
            for i in range(num_images):
                logging.info(
                    "[%s Gradient Leakage Attack %d with %s defense...] Known label is %d.",
                    self.attack_method, trial_number, self.defense_method,
                    torch.argmax(gt_labels[i], dim=-1).item())

        history, losses, mses, lpipss, psnrs, ssims, library_ssims = [], [], [], [], [], [], []
        avg_mses, avg_lpips, avg_psnr, avg_ssim, avg_library_ssim = [], [], [], [], []

        # Conduct gradients/weights/updates matching
        if not self.share_gradients and self.match_weights:
            model = deepcopy(self.trainer.model)
            closure = self.weight_closure(match_optimizer, dummy_data, labels_,
                                          target_weights, model)
        else:
            closure = self.gradient_closure(match_optimizer, dummy_data,
                                            labels_, target_grad)

        for iters in range(num_iters):
            match_optimizer.step(closure)
            current_loss = closure().item()
            losses.append(current_loss)

            if iters % log_interval == 0:
                # Finding evaluation metrics
                eval_dict = get_evaluation_dict(dummy_data, gt_data,
                                                num_images)
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
                    "[%s Gradient Leakage Attack %d with %s defense...] Iter %d: Loss = %.10f, avg MSE = %.8f, avg LPIPS = %.8f, avg PSNR = %.4f dB, avg SSIM = %.3f, avg library SSIM = %.3f",
                    self.attack_method, trial_number, self.defense_method,
                    iters, losses[-1], avg_mses[-1], avg_lpips[-1],
                    avg_psnr[-1], avg_ssim[-1], avg_library_ssim[-1])

                if self.attack_method == 'DLG':
                    history.append([[
                        dummy_data[i].cpu().permute(1, 2, 0).detach().clone(),
                        torch.argmax(dummy_labels[i], dim=-1).item(),
                        dummy_data[i]
                    ] for i in range(num_images)])
                elif self.attack_method == 'iDLG':
                    history.append([[
                        dummy_data[i].cpu().permute(1, 2, 0).detach().clone(),
                        est_labels[i].item(), dummy_data[i]
                    ] for i in range(num_images)])
                elif self.attack_method == 'csDLG':
                    history.append([[
                        dummy_data[i].cpu().permute(1, 2, 0).detach().clone(),
                        torch.argmax(gt_labels[i], dim=-1), dummy_data[i]
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
                csv_processor.write_csv(trial_csv_file, new_row)

        if self.best_mse > avg_mses[-1]:
            self.best_mse = avg_mses[-1]
            self.best_trial = trial_number + 1  # the +1 is because we index from 1 and not 0

        reconstructed_path = f"{trial_result_path}/reconstructed.png"
        self.plot_reconstructed(num_images, history, reconstructed_path)

        # Save the tensors into a .pt file
        tensor_file_path = f"{trial_result_path}/tensors.pt"
        result = {
            i * log_interval: {j: history[i][j][0]
                               for j in range(num_images)}
            for i in range(len(history))
        }
        torch.save(result, tensor_file_path)

        # final_result_path = f"{trial_result_path}/CIFAR2_2-{trial_number+1}.pt"
        # torch.save(history[-1][0][0], final_result_path)

        logging.info("Attack %d complete", (trial_number + 1))

    def gradient_closure(self, match_optimizer, dummy_data, labels,
                         target_grad):
        """ Take a step to match the gradients. """

        def closure():
            match_optimizer.zero_grad()
            try:
                dummy_pred, _ = self.trainer.model(dummy_data)
            except:
                dummy_pred = self.trainer.model(dummy_data)

            dummy_onehot_label = F.softmax(labels, dim=-1)

            if self.attack_method == 'DLG':
                dummy_loss = cross_entropy_for_onehot(dummy_pred,
                                                      dummy_onehot_label)
            elif self.attack_method in ['iDLG', 'csDLG']:
                dummy_loss = cross_entropy(dummy_pred, labels)

            dummy_grad = torch.autograd.grad(dummy_loss,
                                             self.trainer.model.parameters(),
                                             create_graph=True)

            rec_loss = self.reconstruction_costs([dummy_grad], target_grad)
            if hasattr(Config().algorithm, 'total_variation'
                       ) and Config().algorithm.total_variation > 0:
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            return rec_loss

        return closure

    def weight_closure(self, match_optimizer, dummy_data, labels,
                       target_weights, model):
        """ Take a step to match the weights. """

        def closure():
            match_optimizer.zero_grad()
            dummy_weight = self.loss_steps(dummy_data, labels, model)

            rec_loss = self.reconstruction_costs([dummy_weight],
                                                 list(target_weights.values()))
            if hasattr(Config().algorithm, 'total_variation'
                       ) and Config().algorithm.total_variation > 0:
                rec_loss += Config().algorithm.total_variation * TV(dummy_data)
            rec_loss.backward()
            return rec_loss

        return closure

    def loss_steps(self, dummy_data, labels, model):
        """ Take a few gradient descent steps to fit the model to the given input. """
        patched_model = PatchedModule(model)
        if self.use_updates:
            patched_model_origin = deepcopy(patched_model)

        # TODO: optional parameters: lr_schedule, create_graph...
        for epoch in range(epochs):
            if batch_size == 1:
                dummy_pred = patched_model(dummy_data,
                                           patched_model.parameters)
                labels_ = labels
            else:
                idx = epoch % (dummy_data.shape[0] // batch_size)
                dummy_pred = patched_model(
                    dummy_data[idx * batch_size:(idx + 1) * batch_size],
                    patched_model.parameters)
                labels_ = labels[idx * batch_size:(idx + 1) * batch_size]

            loss = cross_entropy(dummy_pred, labels_).sum()

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
            patched_model.parameters = OrderedDict(
                (name, param - param_origin)
                for ((name, param), (name_origin, param_origin)
                     ) in zip(patched_model.parameters.items(),
                              patched_model_origin.parameters.items()))
        return list(patched_model.parameters.values())

    def save_best(self):
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
    def plot_gt(num_images, gt_data, gt_labels):
        """ Plot ground truth data. """
        gt_result_path = f"{dlg_result_path}/ground_truth.png"
        gt_figure = plt.figure(figsize=(12, 4))

        # Make the path if it does not exist yet
        if not os.path.exists(dlg_result_path):
            os.makedirs(dlg_result_path)

        for i in range(num_images):
            current_label = torch.argmax(gt_labels[i], dim=-1).item()
            logging.info("Ground truth labels: %d", current_label)
            gt_figure.add_subplot(1, num_images, i + 1)
            plt.imshow(gt_data[i].cpu().permute(1, 2, 0))
            # torch.save(gt_data[i].cpu().permute(1, 2, 0), "gt3.pt")
            plt.axis('off')
            plt.title("GT image %d\nLabel: %d" % ((i + 1), current_label))
        plt.savefig(gt_result_path)

    @staticmethod
    def plot_reconstructed(num_images, history, reconstructed_result_path):
        """ Plot the reconstructed data. """
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
            outerplot.set_title("Iter=%d" % (i * log_interval))
            outerplot.axis('off')
            fig.add_subplot(outerplot)

            for j in range(num_images):
                innerplot = plt.Subplot(fig, inner[j])
                innerplot.imshow(history[i][j][0])
                innerplot.axis('off')
                fig.add_subplot(innerplot)
        fig.savefig(reconstructed_result_path)
