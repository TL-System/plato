"""
The typical losses for the contrastive self-supervised learning method.




"""
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    # pylint: disable=abstract-method
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, input_tn):
        ctx.save_for_backward(input_tn)
        output = [
            torch.zeros_like(input_tn) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, input_tn)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input_tn, ) = ctx.saved_tensors
        grad_out = torch.zeros_like(input_tn)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NTXent(nn.Module):
    """ The NTXent loss utilized by most self-supervised methods.

        Note: An important issue existed in this implementation
        of NT_Xent as:
        the NT_Xent loss utilized by the SimCLR method sets the defined batch_size
        as the parameter. However, at the end of one epoch, the left samples may smaller than
        the batch_size. This makes the #loaded samples != batch_size.
        Working on criterion that is defined with batch_size but receives loaded
        samples whose size is smaller than the batch size may causes problems.
        drop_last = True can alleviate this issue.
        Currently drop_last is default to be False in Plato.
        Under this case, to avoid this issue, we need to set:
        partition_size / batch_size = integar
        partition_size / pers_batch_size = integar

    """

    def __init__(self, batch_size, temperature, world_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self):
        """ Mask out the correlated samples. """
        batch_size, world_size = self.batch_size, self.world_size
        collected_samples = 2 * batch_size * world_size
        mask = torch.ones((collected_samples, collected_samples), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017),
        we treat the other 2(N - 1) augmented examples within
        a minibatch as negative examples.
        """
        collected_samples = 2 * self.batch_size * self.world_size
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        collected_z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            collected_z = torch.cat(GatherLayer.apply(collected_z), dim=0)

        sim = self.similarity_f(collected_z.unsqueeze(1),
                                collected_z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU
        # gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i),
                                     dim=0).reshape(collected_samples, 1)

        negative_samples = sim[self.mask].reshape(collected_samples, -1)

        labels = torch.zeros(collected_samples).to(
            positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= collected_samples
        return loss


class CrossStopGradientL2loss(nn.Module):
    """ The l2 loss with the stop gradients. """

    def __init__(self):
        super().__init__()

    @staticmethod
    def mean_squared_error(x, y):
        """ Compute the mean square error. """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, outputs):
        """ Compute the l2 loss for the input with the cross stop gradient. """
        (online_pred_one, online_pred_two), (target_proj_one,
                                             target_proj_two) = outputs

        # use the detach mechanism to stop the gradient for target learner
        loss_one = CrossStopGradientL2loss.mean_squared_error(
            online_pred_one, target_proj_two.detach())
        loss_two = CrossStopGradientL2loss.mean_squared_error(
            online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()