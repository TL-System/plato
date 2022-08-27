"""
Obtaining a model from the PyTorch Hub.
"""

import transformers
from plato.config import Config


class Model:
    """The model loaded from HuggingFace based on model head type."""
    @staticmethod
    def get(model_name=None, **kwargs):  # pylint: disable=unused-argument
        """Returns a named model from HuggingFace."""
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }

        config = transformers.AutoConfig.from_pretrained(
            model_name, **config_kwargs)

        if (hasattr(Config().trainer, "model_head_type")):
            automodel = getattr(transformers, Config().trainer.model_head_type)
        else:
            automodel = getattr(transformers, "AutoModelForCausalLM")

        return automodel.from_pretrained(
            model_name,
            config=config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )
