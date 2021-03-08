"""The Feedback Transformer model for PyTorch.

Reference:

Fan, et al. "Addressing Some Limitations of Transformers with Feedback Memory,"
https://arxiv.org/abs/2002.09402.
"""
from feedback_transformer_pytorch import FeedbackTransformer

from models import base
from config import Config


class Model(base.Model, FeedbackTransformer):
    """The Feedback Transformer model for language modeling. """
    @staticmethod
    def is_valid_model_type(model_type):
        return model_type == 'feedback_transformer'

    @staticmethod
    def get_model_from_type(model_type):
        """Obtaining an instance of this model provided that the name is valid."""

        if not Model.is_valid_model_type(model_type):
            raise ValueError('Invalid model type: {}'.format(model_type))

        num_tokens = Config().trainer.num_tokens if hasattr(
            Config().trainer, 'num_tokens') else 20000
        dim = Config().trainer.dim if hasattr(Config().trainer, 'dim') else 512
        depth = Config().trainer.depth if hasattr(Config().trainer,
                                                  'depth') else 6
        seq_len = Config().trainer.seq_len if hasattr(Config().trainer,
                                                      'seq_len') else 2
        mem_len = Config().trainer.mem_len if hasattr(Config().trainer,
                                                      'mem_len') else 256
        dim_head = Config().trainer.dim_head if hasattr(
            Config().trainer, 'dim_head') else 64
        heads = Config().trainer.heads if hasattr(Config().trainer,
                                                  'heads') else 8
        attn_dropout = Config().trainer.attn_dropout if hasattr(
            Config().trainer, 'attn_dropout') else 0.1
        ff_dropout = Config().trainer.ff_dropout if hasattr(
            Config().trainer, 'ff_dropout') else 0.1

        return Model(num_tokens=num_tokens,
                     dim=dim,
                     depth=depth,
                     seq_len=seq_len,
                     mem_len=mem_len,
                     dim_head=dim_head,
                     heads=heads,
                     attn_dropout=attn_dropout,
                     ff_dropout=ff_dropout)
