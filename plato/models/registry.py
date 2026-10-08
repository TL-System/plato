"""
The registry for machine learning models.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

from typing import Any, TypedDict, cast

from plato.config import Config
from plato.models import (
    cnn_encoder,
    dcgan,
    general_multilayer,
    huggingface,
    lenet5,
    multilayer,
    resnet,
    torchvision,
    vgg,
)
from plato.utils.retired_backends import raise_if_retired

_MLX_UNLOADED = object()
mlx_lenet5 = cast(Any, _MLX_UNLOADED)


def _load_mlx_lenet5() -> Any:
    """Load the native model only after an explicit MLX selection."""
    global mlx_lenet5
    if mlx_lenet5 is _MLX_UNLOADED:
        try:
            from plato.models.mlx import lenet5 as native_lenet5
        except ImportError as exc:
            raise ImportError(
                "MLX models require the optional mlx dependency on Apple Silicon."
            ) from exc
        mlx_lenet5 = native_lenet5
        if registered_mlx_models.get("mlx_lenet5") is _mlx_lenet5_model:
            registered_mlx_models["mlx_lenet5"] = native_lenet5.Model
    if mlx_lenet5 is None:
        raise ImportError(
            "MLX models require the optional mlx dependency on Apple Silicon."
        )
    return mlx_lenet5


def _mlx_lenet5_model(**kwargs: Any) -> Any:
    """Keep the built-in factory callable before its native module is loaded."""
    return _load_mlx_lenet5().Model(**kwargs)


registered_models = {
    "lenet5": lenet5.Model,
    "dcgan": dcgan.Model,
    "multilayer": multilayer.Model,
}

registered_factories = {
    "resnet": resnet.Model,
    "vgg": vgg.Model,
    "cnn_encoder": cnn_encoder.Model,
    "general_multilayer": general_multilayer.Model,
    "torchvision": torchvision.Model,
    "huggingface": huggingface.Model,
    "timesfm": huggingface.Model,
    "patchtsmixer": huggingface.Model,
}

registered_mlx_models = {"mlx_lenet5": _mlx_lenet5_model}


class ModelKwargs(TypedDict, total=False):
    model_name: str
    model_type: str
    model_params: dict[str, Any]


def get(**kwargs: Any) -> Any:
    """Get the model with the provided name."""
    config = Config()

    # Get model name
    model_name: str = ""
    if "model_name" in kwargs:
        model_name = cast(str, kwargs["model_name"])
    elif hasattr(config, "trainer"):
        trainer = getattr(config, "trainer")
        if hasattr(trainer, "model_name"):
            model_name = getattr(trainer, "model_name")

    # Get model type
    model_type: str = ""
    if "model_type" in kwargs:
        model_type = cast(str, kwargs["model_type"])
    elif hasattr(config, "trainer"):
        trainer = getattr(config, "trainer")
        if hasattr(trainer, "model_type"):
            model_type = getattr(trainer, "model_type")

    # Get model framework (optional)
    model_framework: str = ""
    if "model_framework" in kwargs:
        model_framework = cast(str, kwargs["model_framework"])
    elif hasattr(config, "trainer"):
        trainer = getattr(config, "trainer")
        if hasattr(trainer, "model_framework"):
            model_framework = getattr(trainer, "model_framework")
        elif hasattr(trainer, "framework"):
            model_framework = getattr(trainer, "framework")

    if not model_framework and hasattr(config, "parameters"):
        parameters = getattr(config, "parameters")
        if hasattr(parameters, "model") and hasattr(parameters.model, "_asdict"):
            model_dict = parameters.model._asdict()
            model_framework = model_dict.get("framework", "")

    # If model_type is still empty, derive it from model_name
    if not model_type and model_name:
        model_type = model_name.split("_")[0]

    raise_if_retired(model_type, category="model")

    # Get model parameters
    model_params: dict[str, Any] = {}
    if "model_params" in kwargs:
        model_params = cast(dict[str, Any], kwargs["model_params"])
    elif hasattr(config, "parameters"):
        parameters = getattr(config, "parameters")
        if hasattr(parameters, "model"):
            model = getattr(parameters, "model")
            if hasattr(model, "_asdict"):
                model_params = model._asdict()

    safe_params = {k: v for k, v in model_params.items() if k != "framework"}

    framework = model_framework.lower()
    mlx_name = model_name.lower().startswith("mlx_")
    if framework == "mlx" or mlx_name:
        if framework and framework != "mlx":
            raise ValueError("MLX model name conflicts with the requested framework.")
        parameter_framework = str(model_params.get("framework", "mlx")).lower()
        if parameter_framework != "mlx":
            raise ValueError(
                "MLX model selection conflicts with parameters.model.framework."
            )
        _load_mlx_lenet5()
        candidate_keys = []
        if model_type:
            candidate_keys.append(f"mlx_{model_type}")
            candidate_keys.append(model_type)
        if model_name:
            candidate_keys.append(model_name)
        for key in candidate_keys:
            key_lower = key.lower()
            if key_lower in registered_mlx_models:
                from plato.trainers.mlx import _resolve_device, _rng_scope, mx

                with (
                    mx.stream(_resolve_device()),
                    _rng_scope(getattr(config.trainer, "model_seed", None)),
                ):
                    model = registered_mlx_models[key_lower](**safe_params)
                    mx.eval(model.state)
                    return model
        raise ValueError(f"No native MLX model registered for: {model_name}")
    if model_type in registered_models:
        registered_model = registered_models[model_type]
        return registered_model(**safe_params)

    if model_type in registered_factories:
        return registered_factories[model_type].get(
            model_name=model_name, **model_params
        )

    raise ValueError(f"No such model: {model_name}")
