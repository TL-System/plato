"""
Federated averaging tailored for LoRA adapters.
"""

from __future__ import annotations

from typing import Optional

from plato.algorithms import fedavg

try:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
except ImportError:  # pragma: no cover
    get_peft_model_state_dict = None  # type: ignore
    set_peft_model_state_dict = None  # type: ignore


class Algorithm(fedavg.Algorithm):
    """FedAvg variant that exchanges only LoRA adapter weights."""

    @staticmethod
    def _peft_base(model) -> Optional[object]:
        """Return the underlying base model that stores LoRA adapters."""
        if model is None:
            return None
        if hasattr(model, "base_model"):
            return model.base_model
        if hasattr(model, "model"):
            return model.model
        return model

    @staticmethod
    def _require_peft():
        if get_peft_model_state_dict is None or set_peft_model_state_dict is None:
            raise ImportError(
                "The 'peft' package is required for LoRA federated training. "
                "Install it by running `uv add peft`."
            )

    def extract_weights(self, model=None):
        """Extract only the LoRA adapter parameters."""
        Algorithm._require_peft()
        peft_base = self._peft_base(model or self.model)
        state_dict = get_peft_model_state_dict(peft_base)
        return {name: tensor.cpu() for name, tensor in state_dict.items()}

    def load_weights(self, weights):
        """Load LoRA adapter parameters into the underlying model."""
        Algorithm._require_peft()
        peft_base = self._peft_base(self.model)
        set_peft_model_state_dict(peft_base, weights)
