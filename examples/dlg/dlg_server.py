"""
An honest-but-curious federated learning server which can
analyze periodic gradients from certain clients to
perform the gradient leakage attacks and
reconstruct the training data of the victim clients.


References:

Zhu et al., "Deep Leakage from Gradients,"
in Advances in Neural Information Processing Systems 2019.

https://papers.nips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

Wang et al., "Protect Privacy from Gradient Leakage Attack in Federated Learning,"
in Proc. INFOCOM 2022.

https://infocom.info/day/2/track/Track%20A#A-3
"""

import logging
from collections import OrderedDict
from copy import deepcopy

import lpips
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from plato.config import Config
from plato.servers import fedavg
from torchvision import transforms

from utils import cross_entropy_for_onehot

criterion = cross_entropy_for_onehot
# TODO: replace hard coded
loss_criterion = torch.nn.CrossEntropyLoss()
tt = transforms.ToPILImage()
loss_fn = lpips.LPIPS(net='vgg')
torch.manual_seed(Config().algorithm.random_seed)


class Server(fedavg.Server):
    """ An honest-but-curious federated learning server with gradient leakage attack. """

    def __init__(self):
        super().__init__()

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
        __, __, payload, __ = updates[Config().algorithm.victim_client]
        # Receive the ground truth for evaluation
        # It will not be used for data reconstruction
        gt_data, gt_label, target_grad = payload[1]
        target_weight = payload[0]

        if not (hasattr(Config().algorithm, 'share_gradients') and Config().algorithm.share_gradients) and \
                not (hasattr(Config().algorithm, 'match_weight') and Config().algorithm.match_weight):
            # Obtain the local updates from clients
            deltas_received = self.compute_weight_deltas(updates)
            target_grad = []
            for delta in deltas_received[
                    Config().algorithm.victim_client].values():
                target_grad.append(-delta / Config().trainer.learning_rate)

        # Plot ground truth data
        partition_size = Config().data.partition_size

        gt_figure = plt.figure(figsize=(12, 4))

        for i in range(partition_size):
            gt_figure.add_subplot(1, partition_size, i + 1)
            plt.imshow(tt(gt_data[i][0].cpu()))
            plt.axis('off')
            plt.title("GT image %d" % (i + 1))

        dummy_data = []
        dummy_label = []
        optimizer = []
        closure = []

        # Generate dummy items
        data_size = self.testset.data[0].shape
        if len(data_size) == 2:
            data_size = (1, 1, data_size[0], data_size[1])
        else:
            data_size = (1, data_size[2], data_size[0], data_size[1])
        for i in range(partition_size):
            dummy_data.append(
                torch.randn(data_size).to(
                    Config().device()).requires_grad_(True))
            dummy_label.append(
                torch.randn((1, Config().trainer.num_classes)).to(
                    Config().device()).requires_grad_(True))
            optimizer.append(torch.optim.LBFGS([dummy_data[i],
                                                dummy_label[i]]))

            logging.info("[Gradient Leakage Attacking...] Dummy label is %d.",
                         torch.argmax(dummy_label[i], dim=-1).item())

        # TODO: periodic analysis, which round?
        # Gradient matching
        history = []
        for i in range(partition_size):
            history.append([])
        losses = []
        mses = []
        lpipss = []

        # Sharing and matching updates
        if not (hasattr(Config().algorithm, 'share_gradients') and Config().algorithm.share_gradients) \
                and hasattr(Config().algorithm, 'match_weight') and Config().algorithm.match_weight:
            for i in range(partition_size):
                model = deepcopy(self.trainer.model)
                closure.append(
                    self.weight_closure(optimizer[i], dummy_data[i],
                                        dummy_label[i], target_weight, model))
        else:
            for i in range(partition_size):
                closure.append(
                    self.gradient_closure(optimizer[i], dummy_data[i],
                                          dummy_label[i], target_grad))

        for iters in range(Config().algorithm.num_iters):
            current = iters % partition_size
            optimizer[current].step(closure[current])
            current_loss = closure[current]().item()
            losses.append(current_loss)
            mses.append(
                torch.mean((dummy_data[current] - gt_data[current])**2).item())
            lpipss.append(
                loss_fn.forward(dummy_data[current], gt_data[current]))

            if iters % Config().algorithm.log_interval == 0:
                # logging.info(
                #     "[Gradient Leakage Attacking...] Iter %d: Loss = %.10f, MSE = %.8f, LPIPS = %.8f",
                #     iters, losses[-1], mses[-1], lpipss[-1])
                for i in range(partition_size):
                    history[i].append(tt(dummy_data[i][0].cpu()))
                logging.info("iter %d", iters)

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
                                                     partition_size,
                                                     subplot_spec=outer[i])
            outerplot = plt.Subplot(fig, outer[i])
            outerplot.set_title("Iter=%d" %
                                (i * Config().algorithm.log_interval))
            outerplot.axis('off')
            fig.add_subplot(outerplot)

            for j in range(partition_size):
                innerplot = plt.Subplot(fig, inner[j])
                innerplot.imshow(history[j][i])
                innerplot.axis('off')
                fig.add_subplot(innerplot)
        # logging.info(
        #     "[Gradient Leakage Attacking...] Reconstructed label is %d.",
        #     torch.argmax(dummy_label, dim=-1).item())
        plt.show()

    def gradient_closure(self, optimizer, dummy_data, dummy_label,
                         target_grad):
        """ Take a step to match the gradients. """

        def closure():
            optimizer.zero_grad()
            # self.trainer.model.zero_grad()
            dummy_pred = self.trainer.model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_grad = torch.autograd.grad(dummy_loss,
                                             self.trainer.model.parameters(),
                                             create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_grad, target_grad):
                grad_diff += ((gx - gy)**2).sum()
            grad_diff.backward()
            return grad_diff

        return closure

    def weight_closure(self, optimizer, dummy_data, dummy_label, target_weight,
                       model):
        """ Take a step to match the model weights. """

        def closure():
            optimizer.zero_grad()
            # self.trainer.model.zero_grad()
            dummy_weight = self.loss_steps(dummy_data, dummy_label, model)

            weight_diff = 0
            for wx, wy in zip(dummy_weight.values(), target_weight.values()):
                weight_diff += ((wx - wy)**2).sum()
            weight_diff.backward()
            return weight_diff

        return closure

    def loss_steps(self, dummy_data, dummy_label, model):
        """ Take a few gradient descent steps to fit the model to the dummy data and labels. """
        epochs = Config().trainer.epochs
        batch_size = Config().trainer.batch_size
        # TODO: use_updates or not
        for epoch in range(epochs):
            if batch_size == 0:
                dummy_pred = model(dummy_data)
                labels_ = dummy_label
            else:
                idx = epoch % (dummy_data.shape[0] // batch_size)
                dummy_pred = model(dummy_data[idx * batch_size:(idx + 1) *
                                              batch_size])
                labels_ = dummy_label[idx * batch_size:(idx + 1) * batch_size]
            dummy_loss = loss_criterion(dummy_pred,
                                        torch.argmax(labels_, dim=-1))
            grad = torch.autograd.grad(dummy_loss,
                                       model.parameters(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)
            with torch.no_grad():
                parameters = OrderedDict(model.named_parameters())
                for (name, param), grad_part in zip(parameters.items(), grad):
                    param -= Config().trainer.learning_rate * grad_part
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data[0])
        #         break
        return OrderedDict(model.named_parameters())
