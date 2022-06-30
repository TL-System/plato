"""
Implementation of the MoCo's network.

"""

import copy
import torch
import torch.nn as nn

from plato.config import Config
from plato.models import encoders_register
from plato.models import general_mlps_register

# utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [
#         torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())
#     ]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output


class FinalMLP(nn.Module):
    """ The implementation of MoCO's prediction layer. """

    def __init__(self, in_dim):
        super().__init__()

        self.layers = general_mlps_register.Model.get_model(
            model_type="moco_final_mlp", input_dim=in_dim)

    def forward(self, x):
        """ Forward the projection layer. """
        for layer in self.layers:
            x = layer(x)

        return x

    def output_dim(self):
        """ Obtain the output dimension. """
        return self.layers[-1][-1].out_features


class EncoderwithMLP(nn.Module):
    """ The module combining the encoder and the projection. """

    def __init__(self, encoder=None, encode_dim=None):
        super().__init__()

        # define the encode based on the model_name in config
        if encoder is None:
            self.encoder, self.encode_dim = encoders_register.get()
        # utilize the custom model
        else:
            self.encoder, self.encode_dim = encoder, encode_dim

        self.final_mlp = FinalMLP(in_dim=self.encode_dim)

        self.final_mlp_dim = self.final_mlp.output_dim()

    def forward(self, x):
        """ Forward the encoder and the projection """
        x = self.encoder(x)
        x = self.final_mlp(x)
        return x

    def get_final_dim(self):
        return self.final_mlp_dim


class MoCo(nn.Module):
    """ The implementation of SimSiam method. """

    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # define the encoders based on the model_name in config
        if encoder is None:
            self.encoder_q = EncoderwithMLP()
            self.encoder_k = EncoderwithMLP()
            self.encoder_q_dim = self.encoder_q.encode_dim
            self.encoder_k_dim = self.encoder_k.encode_dim
        # utilize the custom model
        else:
            self.encoder_q, self.encoder_q_dim = copy.deepcopy(
                encoder), encoder_dim
            self.encoder_k, self.encoder_k_dim = copy.deepcopy(
                encoder), encoder_dim

        self._initial_k_with_q()
        # obtain the necessary hyper-parameters
        feature_dim = self.encoder_q.get_final_dim()

        # the number of negative keys, K (default: 65536)
        self.queue_size = Config().trainer.queue_size
        # moco momentum of updating key encoder (default: 0.999)
        self.update_momentum = Config().trainer.update_momentum
        # softmax temperature (default: 0.07)
        self.temperature = Config().trainer.temperature

        # create the queue
        self.register_buffer("queue", torch.randn(feature_dim,
                                                  self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _initial_k_with_q(self):

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.update_momentum + param_q.data * (
                1. - self.update_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()

    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)

    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)

    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this], idx_unshuffle

    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this]

    def forward(self, augmented_samples):
        """

        Output:
            logits, targets
        """

        # im_q: a batch of query images
        # im_k: a batch of key images
        im_q, im_k = augmented_samples

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0],
                             device=im_q.device,
                             dtype=torch.long)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    @property
    def encoder(self):
        """ Obtain the Moco's encoder """
        return self.encoder_q.encoder

    @property
    def encode_dim(self):
        """ Obtain the  encoder. """
        return self.encoder_q.encode_dim

    @staticmethod
    def get_model():
        """Obtaining an instance of this model provided that the name is valid."""

        return MoCo()