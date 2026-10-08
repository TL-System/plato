"""MLX selection must diagnose incompatible models and backend choices."""

import pytest

from plato.algorithms import registry as algorithms
from plato.models import registry as models
from plato.trainers import registry as trainers
from tests.integration.utils import configure_environment
from tests.mlx_native.helpers import configuration


@pytest.mark.parametrize(
    "registry,section,value",
    [
        (models, "model", "resnet18"),
        (trainers, "trainer", "basic"),
        (algorithms, "algorithm", "fedavg"),
    ],
    ids=[
        "unsupported-native-model",
        "torch-trainer-with-mlx",
        "torch-algorithm-with-mlx",
    ],
)
def test_explicit_mlx_cannot_fall_back(tmp_path, registry, section, value):
    config = configuration()
    if section == "model":
        config["trainer"]["model_name"] = value
    else:
        config[section]["type"] = value
    with configure_environment(config, runtime_root=tmp_path):
        with pytest.raises(ValueError, match="MLX|mlx"):
            registry.get()


def test_missing_native_model_dependency_diagnosed(tmp_path, monkeypatch):
    with configure_environment(configuration(), runtime_root=tmp_path):
        monkeypatch.setattr(models, "mlx_lenet5", None)
        monkeypatch.setattr(models, "registered_mlx_models", {})
        with pytest.raises(ImportError, match="MLX|mlx"):
            models.get()


def test_conflicting_model_framework_diagnosed(tmp_path):
    config = configuration()
    config["parameters"]["model"]["framework"] = "torch"
    with configure_environment(config, runtime_root=tmp_path):
        with pytest.raises(ValueError, match="framework"):
            models.get()


def test_native_model_name_cannot_override_explicit_torch_backend(tmp_path):
    with configure_environment(configuration(), runtime_root=tmp_path):
        with pytest.raises(ValueError, match="MLX.*framework"):
            models.get(
                model_name="mlx_lenet5",
                model_framework="torch",
                model_params={"num_classes": 10},
            )


def test_native_model_name_shortcut_scopes_factory_to_cpu(tmp_path, monkeypatch):
    import mlx.core as mx

    from plato.models.mlx.lenet5 import LeNet5

    config = configuration()
    config["trainer"].pop("framework")
    config["parameters"]["model"].pop("framework")
    observed = []

    def factory(**kwargs):
        observed.append(mx.default_device())
        return LeNet5(**kwargs)

    with configure_environment(config, runtime_root=tmp_path):
        monkeypatch.setitem(models.registered_mlx_models, "mlx_lenet5", factory)
        models.get(model_name="mlx_lenet5")
        assert observed == [mx.cpu]
