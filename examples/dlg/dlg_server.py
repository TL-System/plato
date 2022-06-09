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
import logging
import math
import os
import sys
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

from utils.modules import MetaMonkey
from utils.utils import cross_entropy_for_onehot

cross_entropy = torch.nn.CrossEntropyLoss()
tt = transforms.ToPILImage()
loss_fn = lpips.LPIPS(net='vgg')
torch.manual_seed(Config().algorithm.random_seed)

dlg_result_path = f"{Config().params['result_path']}"
dlg_result_file = f"{dlg_result_path}/{os.getpid()}_evals.csv"
dlg_result_headers = [
    "Iteration", "Loss", "Average MSE", "Average LPIPS"
]
csv_processor.initialize_csv(dlg_result_file, dlg_result_headers,
                             dlg_result_path)


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

    async def process_reports(self):
        """ Process the client reports: before aggregating their weights,
            perform the gradient leakage attacks and reconstruct the training data.
        """
        if self.current_round == Config().algorithm.attack_round:
            self.deep_leakage_from_gradients(self.updates)
        await self.aggregate_weights(self.updates)

    def compute_weight_deltas(self, updates):
        """ Extract the model weight updates from client updates. """
        weights_received = [payload[0] for (__, __, payload, __) in updates]
        return self.algorithm.compute_weight_deltas(weights_received)

    def deep_leakage_from_gradients(self, updates):
        """ Analyze periodic gradients from certain clients. """
        # Process data from the victim client
        __, __, payload, __ = updates[Config().algorithm.victim_client]
        # The ground truth should be used only for evaluation
        gt_data, gt_label, target_grad = payload[1]
        target_weight = payload[0]

        # Assume the reconstructed data shape is known, which can be also derived from the target dataset
        num_images = Config().data.partition_size
        data_size = [num_images, gt_data.shape[1],
                     gt_data.shape[2], gt_data.shape[3]]
        self.plot_gt(num_images, gt_data, gt_label)

        if not (hasattr(Config().algorithm, 'share_gradients') and Config().algorithm.share_gradients) and \
                not (hasattr(Config().algorithm, 'match_weight') and Config().algorithm.match_weight):
            # Obtain the local updates from clients
            deltas_received = self.compute_weight_deltas(updates)
            target_grad = []
            for delta in deltas_received[
                    Config().algorithm.victim_client].values():
                target_grad.append(-delta / Config().trainer.learning_rate)

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
                logging.info("[%s Gradient Leakage Attacking...] Dummy label is %d.",
                             self.attack_method, torch.argmax(dummy_label[i], dim=-1).item())
        elif self.attack_method == 'iDLG':
            match_optimizer = torch.optim.LBFGS(
                [dummy_data, ], lr=Config().algorithm.lr)
            # Estimate the gt label
            # TODO: why -2?
            est_label = torch.argmin(torch.sum(
                target_grad[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            for i in range(num_images):
                logging.info("[%s Gradient Leakage Attacking...] Estimated label is %d.",
                             self.attack_method, est_label.item())

        history, losses, mses, avg_mses, lpipss, avg_lpips = [], [], [], [], [], []

        # Conduct gradients/weights/updates matching
        if not (hasattr(Config().algorithm, 'share_gradients') and Config().algorithm.share_gradients) \
                and hasattr(Config().algorithm, 'match_weight') and Config().algorithm.match_weight:
            model = deepcopy(self.trainer.model)
            closure = self.weight_closure(match_optimizer, dummy_data,
                                          dummy_label, target_weight, model)
        else:
            closure = self.gradient_closure(match_optimizer, dummy_data,
                                            dummy_label, est_label, target_grad)

        for iters in range(Config().algorithm.num_iters):
            match_optimizer.step(closure)
            current_loss = closure().item()
            losses.append(current_loss)
            mses.append([])
            lpipss.append([])
            for i in range(num_images):
                mses[iters].append(math.inf)
                lpipss[iters].append(math.inf)
                # Find the closest ground truth data after the misordering
                for j in range(num_images):
                    mses[iters][i] = min(
                        mses[iters][i],
                        torch.mean((dummy_data[i] - gt_data[j])**2).item())
                    lpipss[iters][i] = min(
                        lpipss[iters][i],
                        loss_fn.forward(dummy_data[i], gt_data[j]).item())

            avg_mses.append(mean(mses[iters]))
            avg_lpips.append(mean(lpipss[iters]))

            if iters % Config().algorithm.log_interval == 0:
                logging.info(
                    "[%s Gradient Leakage Attacking...] Iter %d: Loss = %.10f, avg MSE = %.8f, avg LPIPS = %.8f",
                    self.attack_method, iters, losses[-1], avg_mses[-1], avg_lpips[-1])
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
                    round(avg_lpips[-1], 8)
                ]
                csv_processor.write_csv(dlg_result_file, new_row)

        self.plot_reconstructed(num_images, history)

        self.write_history(num_images, history)

        logging.info("Attack complete")

    def gradient_closure(self, match_optimizer, dummy_data, dummy_label, est_label,
                         target_grad):
        """ Take a step to match the gradients. """

        def closure():
            match_optimizer.zero_grad()
            # self.trainer.model.zero_grad()
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

            grad_diff = 0
            for gx, gy in zip(dummy_grad, target_grad):
                grad_diff += ((gx - gy)**2).sum()
            grad_diff.backward()
            return grad_diff

        return closure

    def weight_closure(self, match_optimizer, dummy_data, dummy_label,
                       target_weight, model):
        """ Take a step to match the weights. """

        def closure():
            match_optimizer.zero_grad()
            dummy_weight = self.loss_steps(dummy_data, dummy_label, model)

            weight_diff = 0
            for wx, wy in zip(dummy_weight, target_weight.values()):
                weight_diff += ((wx - wy)**2).sum()
            weight_diff.backward()
            return weight_diff

        return closure

    def loss_steps(self, dummy_data, dummy_label, model):
        """ Take a few gradient descent steps to fit the model to the given input. """
        patched_model = MetaMonkey(model)

        epochs = Config().trainer.epochs
        batch_size = Config().trainer.batch_size

        # TODO: optional parameters: lr_schedule, create_graph...

        # TODO: use updates or weights

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print("before loss steps", param.data[0])
        #         break

        for epoch in range(epochs):
            if batch_size == 1:
                dummy_pred = model(dummy_data)
                labels_ = dummy_label
            else:
                # TODO: local steps vs. epochs
                idx = epoch % (dummy_data.shape[0] // batch_size)
                dummy_pred = model(dummy_data[idx * batch_size:(idx + 1) *
                                              batch_size])
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

        return list(patched_model.parameters.values())

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
        outer = gridspec.GridSpec(
            (Config().algorithm.num_iters // Config().algorithm.log_interval)
            // 2,
            2,
            wspace=0.2,
            hspace=0.2)

        for i in range(Config().algorithm.num_iters //
                       Config().algorithm.log_interval):
            inner = gridspec.GridSpecFromSubplotSpec(1,
                                                     num_images,
                                                     subplot_spec=outer[i])
            outerplot = plt.Subplot(fig, outer[i])
            outerplot.set_title("Iter=%d" %
                                (i * Config().algorithm.log_interval))
            outerplot.axis('off')
            fig.add_subplot(outerplot)

            for j in range(num_images):
                innerplot = plt.Subplot(fig, inner[j])
                innerplot.imshow(history[i][j][0])
                innerplot.axis('off')
                fig.add_subplot(innerplot)
        fig.savefig(reconstructed_result_path)

    @staticmethod
    def write_history(num_images, history):
        """ Write the history of the tensors into text file. """
        log_interval = Config().algorithm.log_interval
        file_path = f"{Config().params['result_path']}/{os.getpid()}_history.txt"
        with open(file_path, 'w') as file:
            for iter in range(len(history)):
                file.write("Iteration: " + str(log_interval * iter) + "\n")
                for img_num in range(num_images):
                    file.write("Image number: " + str(img_num + 1) + "\n")
                    file.write(str(history[iter][img_num][2]) + "\n")
