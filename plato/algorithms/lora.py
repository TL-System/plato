"""
Federated averaging tailored for LoRA adapters.
"""

from __future__ import annotations

from plato.algorithms import fedavg
from plato.utils.huggingface import adapter_save_embeddings

try:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
except ImportError:  # pragma: no cover
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None


class Algorithm(fedavg.Algorithm):
    """FedAvg variant that exchanges only LoRA adapter weights."""

    @staticmethod
    def _peft_base(model) -> object | None:
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
        getter = get_peft_model_state_dict
        setter = set_peft_model_state_dict
        if getter is None or setter is None:
            raise ImportError(
                "The 'peft' package is required for LoRA federated training. "
                "Install it by running `uv add peft`."
            )
        return getter, setter

    def extract_weights(self, model=None):
        """Extract only the LoRA adapter parameters."""
        getter, _ = Algorithm._require_peft()
        peft_base = self._peft_base(model or self.model)
        state_dict = getter(
            peft_base,
            save_embedding_layers=adapter_save_embeddings(model or self.model),
        )
        return {
            name: self._to_transport_tensor(tensor, name)
            for name, tensor in state_dict.items()
        }

    def load_weights(self, weights):
        """Load LoRA adapter parameters into the underlying model."""
        _, setter = Algorithm._require_peft()
        peft_base = self._peft_base(self.model)
        setter(peft_base, weights)
