"""Native MLX test data, configuration and independent numerical assertions."""

from types import SimpleNamespace

import numpy as np

from plato.config import Config
from tests.integration.utils import configure_environment


def configuration(**trainer_options):
    return {
        "clients": {
            "type": "simple",
            "total_clients": 2,
            "per_round": 2,
            "do_test": False,
        },
        "server": {"address": "127.0.0.1", "port": 8000, "do_test": False},
        "data": {
            "datasource": "MNIST",
            "partition_size": 8,
            "sampler": "iid",
            "random_seed": 1,
        },
        "trainer": {
            "type": "mlx",
            "framework": "mlx",
            "rounds": 1,
            "epochs": 1,
            "batch_size": 8,
            "optimizer": "sgd",
            "model_name": "lenet5",
            **trainer_options,
        },
        "algorithm": {"type": "mlx_fedavg", "framework": "mlx"},
        "parameters": {
            "model": {"framework": "mlx", "num_classes": 10},
            "optimizer": {"learning_rate": 0.01},
        },
    }


def native_config(tmp_path, **options):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return configure_environment(configuration(**options), runtime_root=tmp_path)


def dataset(count=8, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, 28, 28, 1)).astype(np.float32)
    y = rng.integers(0, 10, size=count, dtype=np.int32)
    return list(zip(x, y))


def assert_tree_equal(actual, expected, *, rtol=0, atol=0):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_tree_equal(left, right, rtol=rtol, atol=atol)
    elif expected is None:
        assert actual is None
    else:
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def update(client_id, count, weights):
    return SimpleNamespace(
        client_id=client_id,
        payload=weights,
        report=SimpleNamespace(
            num_samples=count,
            accuracy=0.0,
            training_time=0.0,
            processing_time=0.0,
            comm_time=0.0,
            type="weights",
        ),
    )


def no_device_flags():
    Config.args.cpu = False
    Config.args.mps = False
