"""
Obtaining a model from the PyTorch Hub.
"""

from transformers import AutoModelForCausalLM, AutoConfig
from plato.config import Config


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

        return AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )