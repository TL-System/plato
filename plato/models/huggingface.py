"""
Obtaining a model from HuggingFace with optional parameter-efficient fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from transformers import AutoConfig, AutoModelForCausalLM

from plato.config import Config

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - handled at runtime with friendly message.
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


def _lora_config_dict(lora_config: Any) -> Dict[str, Any]:
    """Convert various config objects (namedtuple, SimpleNamespace, dict)."""
    if lora_config is None:
        return {}
    if isinstance(lora_config, dict):
        return dict(lora_config)
    if hasattr(lora_config, "_asdict"):
        return dict(lora_config._asdict())
    if hasattr(lora_config, "__dict__"):
        return {
            key: value
            for key, value in lora_config.__dict__.items()
            if not key.startswith("_")
        }
    raise TypeError("Unsupported LoRA configuration format.")


class Model:
    """The CausalLM model loaded from HuggingFace."""

    @staticmethod
    def get(model_name=None, **kwargs):  # pylint: disable=unused-argument
        """Returns a named model from HuggingFace."""
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }

        config = AutoConfig.from_pretrained(model_name, **config_kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )

        lora_params = getattr(getattr(Config(), "parameters", None), "lora", None)
        if lora_params is not None:
            if get_peft_model is None or LoraConfig is None:
                raise ImportError(
                    "The 'peft' package is required for LoRA fine-tuning. "
                    "Install it by running `uv add peft`."
                )

            params_dict = _lora_config_dict(lora_params)
            logging.info("Configuring LoRA with parameters: %s", params_dict)
            lora_cfg = LoraConfig(**params_dict)
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        if hasattr(model, "loss_type"):
            model.loss_type = "ForCausalLM"

        return model
