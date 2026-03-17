"""
Obtaining a model from HuggingFace with optional parameter-efficient fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    from transformers import TimesFmConfig, TimesFmModelForPrediction
except ImportError:
    TimesFmConfig = None
    TimesFmModelForPrediction = None

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


class _TimesFmOutput:
    """Output container compatible with the Plato time-series training/testing pipeline."""

    def __init__(self, loss=None, prediction_outputs=None):
        self.loss = loss
        self.prediction_outputs = prediction_outputs


class TimesFmMultivariateWrapper(nn.Module):
    """Wraps TimesFmModelForPrediction for batched, multivariate time series.

    TimesFM is natively univariate (each call takes a list of 1-D tensors).
    This wrapper accepts a standard batched tensor of shape
    ``(batch, context_length)``  or  ``(batch, context_length, channels)``
    and handles the reshaping transparently so the rest of the Plato pipeline
    (collators, training/testing strategies) needs no changes.

    For multivariate input, every channel is processed independently through
    the same TimesFM model (channel-independent forecasting).  The outputs are
    recombined into ``(batch, prediction_length, channels)``.

    If ``future_values`` is provided the wrapper computes MSE loss against it
    and stores it in ``.loss``.  ``prediction_outputs`` always holds the mean
    predictions in ``(batch, prediction_length, out_channels)`` form.

    Args:
        model: An instantiated ``TimesFmModelForPrediction``.
        prediction_length: Number of future steps to keep.  Predictions are
            truncated to this length when the model's ``horizon_length``
            differs from the configured ``prediction_length``.
        default_freq: Default frequency token (0 = high/hourly,
            1 = medium/daily-weekly, 2 = low/monthly-yearly).
    """

    def __init__(
        self,
        model: "TimesFmModelForPrediction",
        prediction_length: int | None = None,
        default_freq: int = 0,
    ):
        super().__init__()
        self.model = model
        self.prediction_length = prediction_length
        self.default_freq = default_freq

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: torch.Tensor | None = None,
        freq: int | list | torch.Tensor | None = None,
        return_dict: bool = True,  # accepted for API compat, ignored internally
        **kwargs,
    ) -> _TimesFmOutput:
        if not isinstance(past_values, torch.Tensor):
            raise TypeError("past_values must be a torch.Tensor")

        if past_values.dim() == 3:
            #  Multivariate path
            batch, ctx, channels = past_values.shape
            # (batch, ctx, ch) -> (batch*ch, ctx)
            pv_2d = past_values.permute(0, 2, 1).reshape(batch * channels, ctx)
            past_list = [pv_2d[i] for i in range(pv_2d.size(0))]
            freq_list = self._build_freq_list(freq, batch, channels)

            outputs = self.model(past_values=past_list, freq=freq_list)

            # (batch*ch, horizon) -> (batch, horizon, ch)
            raw = outputs.mean_predictions
            horizon = raw.shape[-1]
            mean_preds = raw.reshape(batch, channels, horizon).permute(0, 2, 1)

        else:
            #  Univariate path
            batch = past_values.size(0)
            past_list = [past_values[i] for i in range(batch)]
            freq_list = self._build_freq_list(freq, batch, channels=1)

            outputs = self.model(past_values=past_list, freq=freq_list)
            mean_preds = outputs.mean_predictions.unsqueeze(-1)  # (batch, horizon, 1)

        # Truncate to configured prediction_length
        if self.prediction_length is not None:
            mean_preds = mean_preds[:, : self.prediction_length, :]

        # Compute MSE loss when targets are provided
        loss = None
        if future_values is not None:
            fv = future_values
            if fv.dim() == 2:
                fv = fv.unsqueeze(-1)  # (batch, pred) -> (batch, pred, 1)
            min_len = min(mean_preds.shape[1], fv.shape[1])
            loss = F.mse_loss(
                mean_preds[:, :min_len, : fv.shape[-1]],
                fv[:, :min_len, :],
            )

        return _TimesFmOutput(loss=loss, prediction_outputs=mean_preds)

    def _build_freq_list(
        self,
        freq: int | list | torch.Tensor | None,
        batch: int,
        channels: int,
    ) -> list[int]:
        n = batch * channels
        if freq is None:
            return [self.default_freq] * n
        if isinstance(freq, int):
            return [freq] * n
        if isinstance(freq, torch.Tensor):
            freq = freq.tolist()
        # freq is list of length batch; expand for each channel
        return [int(f) for f in freq for _ in range(channels)]


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

    # PatchTSMixer

    @staticmethod
    def _get_patchtsmixer_model(resolved_model_name, cache_dir, model_task=None):
        """Load or create a PatchTSMixer model."""
        if PatchTSMixerForPrediction is None:
            raise ImportError(
                "PatchTSMixer models are not available. "
                "Ensure you have transformers>=4.35.0 installed."
            )

        task_type = Model._get_timeseries_task_type(model_task)

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
            logging.info(
                "Model '%s' not found as pretrained, creating from config settings",
                resolved_model_name,
            )
            trainer_config = Config().trainer

            scaling_param = getattr(trainer_config, "scaling", "std")
            if isinstance(scaling_param, str) and scaling_param.lower() == "none":
                scaling_param = None

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
                scaling=scaling_param,
                prediction_channel_indices=getattr(
                    trainer_config, "prediction_channel_indices", None
                ),
            )

            if task_type == "classification":
                config.num_labels = getattr(trainer_config, "num_classes", 2)
                model = PatchTSMixerForTimeSeriesClassification(config)
            elif task_type == "regression":
                config.num_targets = getattr(trainer_config, "num_targets", 1)
                model = PatchTSMixerForRegression(config)
            elif task_type == "pretraining":
                model = PatchTSMixerForPretraining(config)
            else:
                model = PatchTSMixerForPrediction(config)

        return model

    # TimesFM

    @staticmethod
    def _get_timesfm_model(resolved_model_name, cache_dir):
        """Load or create a TimesFM model wrapped for batched multivariate use."""
        if TimesFmModelForPrediction is None:
            raise ImportError(
                "TimesFM models are not available. "
                "Ensure you have transformers>=5.0.0 installed."
            )

        trainer_config = Config().trainer
        prediction_length = getattr(trainer_config, "prediction_length", 128)
        default_freq = getattr(trainer_config, "freq", 0)

        try:
            logging.info(
                "Attempting to load pretrained TimesFM model: %s",
                resolved_model_name,
            )
            inner = TimesFmModelForPrediction.from_pretrained(
                resolved_model_name, cache_dir=cache_dir
            )
            logging.info("Successfully loaded pretrained TimesFM model")
        except (OSError, ValueError, Exception):
            logging.info(
                "TimesFM model '%s' not found as pretrained, creating from config",
                resolved_model_name,
            )
            context_length = getattr(trainer_config, "context_length", 512)
            horizon_length = prediction_length

            config = TimesFmConfig(
                context_length=context_length,
                horizon_length=horizon_length,
                patch_length=getattr(trainer_config, "patch_length", 32),
                num_hidden_layers=getattr(trainer_config, "num_hidden_layers", 20),
                hidden_size=getattr(trainer_config, "hidden_size", 1280),
                intermediate_size=getattr(trainer_config, "intermediate_size", 1280),
                num_attention_heads=getattr(trainer_config, "num_attention_heads", 16),
                head_dim=getattr(trainer_config, "head_dim", 80),
                attention_dropout=getattr(trainer_config, "dropout", 0.0),
            )
            inner = TimesFmModelForPrediction(config)

        return TimesFmMultivariateWrapper(
            model=inner,
            prediction_length=prediction_length,
            default_freq=default_freq,
        )

    # Main factory entry point

    @staticmethod
    def get(model_name=None, **kwargs):  # pylint: disable=unused-argument
        """Returns a named model from HuggingFace."""
        resolved_model_name = (
            model_name
            if isinstance(model_name, str) and model_name
            else getattr(getattr(Config(), "trainer", None), "model_name", None)
        )
        if not isinstance(resolved_model_name, str) or not resolved_model_name:
            raise ValueError("A valid HuggingFace model name must be provided.")

        cache_dir = Config().params["model_path"] + "/huggingface"

        model_type = kwargs.get("model_type") or getattr(
            getattr(Config(), "trainer", None), "model_type", None
        )

        is_timeseries = is_timeseries_model(
            model_name=resolved_model_name, model_type=model_type
        )

        if is_timeseries:
            model_type_lower = (model_type or "").lower()
            model_name_lower = resolved_model_name.lower()

            if model_type_lower == "timesfm" or "timesfm" in model_name_lower:
                return Model._get_timesfm_model(resolved_model_name, cache_dir)

            # Default time-series path -> PatchTSMixer
            model_task = kwargs.get("model_task")
            return Model._get_patchtsmixer_model(
                resolved_model_name, cache_dir, model_task
            )

        #  NLP / CausalLM path
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }
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
