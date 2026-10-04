"""
A split learning algorithm supporting LoRA fine-tuning LLMs.
"""

from typing import Dict, cast

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import Tensor
from torch.nn import Module

from plato.algorithms import split_learning


class Algorithm(split_learning.Algorithm):
    """
    Extract and load only the LoRA weights.
    """

    def _get_base_model(self, model: object | None = None) -> Module:
        """Return the wrapped HuggingFace base model."""
        model_obj = model if model is not None else self.model
        if model_obj is None or not hasattr(model_obj, "base_model"):
            raise AttributeError(
                "LoRA split learning requires a model with a `base_model` attribute."
            )
        base_model = getattr(model_obj, "base_model")
        return cast(Module, base_model)

    def extract_weights(self, model=None) -> Dict[str, Tensor]:
        """Extract LoRA weights from the underlying base model."""
        base_model = self._get_base_model(model)
        return {k: v.cpu() for k, v in get_peft_model_state_dict(base_model).items()}

    def load_weights(self, weights: Dict[str, Tensor]):
        """Load LoRA weights into the underlying base model."""
        base_model = self._get_base_model()
        return set_peft_model_state_dict(base_model, weights)
