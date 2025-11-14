"""
Obtaining a model from HuggingFace with optional parameter-efficient fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from transformers import AutoConfig, AutoModelForCausalLM

from plato.config import Config
from plato.utils.timeseries_utils import is_timeseries_model

try:
    from transformers import (
        PatchTSMixerConfig,
        PatchTSMixerForPrediction,
        PatchTSMixerForPretraining,
        PatchTSMixerForRegression,
        PatchTSMixerForTimeSeriesClassification,
    )
except ImportError:
    PatchTSMixerConfig = None
    PatchTSMixerForPrediction = None
    PatchTSMixerForTimeSeriesClassification = None
    PatchTSMixerForRegression = None
    PatchTSMixerForPretraining = None

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - handled at runtime with friendly message.
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


def _lora_config_dict(lora_config: Any) -> dict[str, Any]:
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
    """The HuggingFace model factory supporting various model types."""

    @staticmethod
    def _get_timeseries_task_type(model_task=None):
        """Determine the task type for time series models from config or arguments."""
        trainer_config = Config().trainer
        return (
            model_task
            or getattr(trainer_config, "model_task", None)
            or getattr(trainer_config, "task_type", "forecasting")
        )

    @staticmethod
    def _get_patchtsmixer_model(resolved_model_name, cache_dir, model_task=None):
        """Load or create a PatchTSMixer model."""
        if PatchTSMixerForPrediction is None:
            raise ImportError(
                "PatchTSMixer models are not available. "
                "Ensure you have transformers>=4.35.0 installed."
            )

        task_type = Model._get_timeseries_task_type(model_task)

        # Try to load pretrained model first
        task_models = {
            "classification": PatchTSMixerForTimeSeriesClassification,
            "regression": PatchTSMixerForRegression,
            "pretraining": PatchTSMixerForPretraining,
            "forecasting": PatchTSMixerForPrediction,
        }
        model_class = task_models.get(task_type, PatchTSMixerForPrediction)

        try:
            logging.info(
                "Attempting to load pretrained PatchTSMixer model: %s",
                resolved_model_name,
            )
            model = model_class.from_pretrained(
                resolved_model_name, cache_dir=cache_dir
            )
            logging.info("Successfully loaded pretrained model")
        except (OSError, ValueError, Exception):
            # If loading fails, create new model from config
            logging.info(
                "Model '%s' not found as pretrained, creating from config settings",
                resolved_model_name,
            )
            trainer_config = Config().trainer

            config = PatchTSMixerConfig(
                context_length=getattr(trainer_config, "context_length", 512),
                prediction_length=getattr(trainer_config, "prediction_length", 96),
                num_input_channels=getattr(trainer_config, "num_input_channels", 7),
                patch_length=getattr(trainer_config, "patch_length", 8),
                patch_stride=getattr(trainer_config, "patch_stride", 8),
                d_model=getattr(trainer_config, "d_model", 64),
                num_layers=getattr(trainer_config, "num_layers", 8),
                expansion_factor=getattr(trainer_config, "expansion_factor", 2),
                dropout=getattr(trainer_config, "dropout", 0.2),
                head_dropout=getattr(trainer_config, "head_dropout", 0.2),
                mode=getattr(trainer_config, "mode", "common_channel"),
                gated_attn=getattr(trainer_config, "gated_attn", True),
                scaling=getattr(trainer_config, "scaling", "std"),
                prediction_channel_indices=getattr(
                    trainer_config, "prediction_channel_indices", None
                ),
            )

            # Set task-specific parameters and create model
            if task_type == "classification":
                config.num_labels = getattr(trainer_config, "num_classes", 2)
                model = PatchTSMixerForTimeSeriesClassification(config)
            elif task_type == "regression":
                config.num_targets = getattr(trainer_config, "num_targets", 1)
                model = PatchTSMixerForRegression(config)
            elif task_type == "pretraining":
                model = PatchTSMixerForPretraining(config)
            else:  # forecasting
                model = PatchTSMixerForPrediction(config)

        return model

    @staticmethod
    def get(model_name=None, **kwargs):  # pylint: disable=unused-argument
        """Returns a named model from HuggingFace."""
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }

        resolved_model_name = (
            model_name
            if isinstance(model_name, str) and model_name
            else getattr(getattr(Config(), "trainer", None), "model_name", None)
        )
        if not isinstance(resolved_model_name, str) or not resolved_model_name:
            raise ValueError("A valid HuggingFace model name must be provided.")

        cache_dir = Config().params["model_path"] + "/huggingface"

        # Determine model type from config or model name
        model_type = kwargs.get("model_type") or getattr(
            getattr(Config(), "trainer", None), "model_type", None
        )

        # Detect if this is a time series model and which type
        is_timeseries = is_timeseries_model(
            model_name=resolved_model_name, model_type=model_type
        )

        if is_timeseries:
            model_task = kwargs.get("model_task")
            return Model._get_patchtsmixer_model(
                resolved_model_name, cache_dir, model_task
            )

        # Default to CausalLM for backward compatibility
        config = AutoConfig.from_pretrained(resolved_model_name, **config_kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
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
            setattr(model, "loss_type", "ForCausalLM")

        return model
