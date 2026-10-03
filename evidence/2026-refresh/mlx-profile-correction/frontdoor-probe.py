import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from plato.algorithms import registry as algorithms
from plato.models import registry as models
from plato.serialization.safetensor import deserialize_tree, serialize_tree
from plato.utils import tree
from tests.conftest import _native_prerequisites

output = Path(sys.argv[1])
assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)
assert 'mlx_fedavg' in algorithms.registered_algorithms
assert 'mlx_lenet5' in models.registered_mlx_models
try:
    version = importlib.metadata.version('mlx')
except importlib.metadata.PackageNotFoundError:
    version = None
record = {'mlx_distribution_version': version, 'backend_modules_before_explicit_selection': [], 'builtin_native_registry_keys_preserved': True}
if version is None:
    config = SimpleNamespace(trainer=SimpleNamespace(model_name='lenet5', framework='mlx'))
    models.Config = lambda: config
    try:
        models.get()
    except ImportError as exc:
        assert 'optional mlx dependency' in str(exc)
        assert isinstance(exc.__cause__, ImportError)
        record['model_missing_diagnostic'] = str(exc)
    else:
        raise AssertionError('missing native dependency model must fail')
    algorithms.Config = lambda: SimpleNamespace(algorithm=SimpleNamespace(type='mlx_fedavg', framework='mlx'))
    try:
        algorithms.get()
    except ImportError as exc:
        assert 'MLX is not installed' in str(exc)
        assert isinstance(exc.__cause__, ImportError)
        record['algorithm_missing_diagnostic'] = str(exc)
    else:
        raise AssertionError('missing native dependency algorithm must fail')
else:
    _native_prerequisites()
    import mlx.core as mx
    from plato.algorithms.mlx_fedavg import Algorithm
    from plato.models.mlx.lenet5 import LeNet5, Model
    from plato.trainers.mlx import ComposableMLXTrainer
    from tests.integration.utils import configure_environment
    from tests.mlx_native.helpers import configuration
    native = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    source = {'native': native, 'none': None, 'sequence': (np.array(2.0), torch.tensor([1, 2], dtype=torch.bfloat16)), 'text': 'wire'}
    flat, metadata = tree.flatten_tree(source)
    assert metadata['native'].backend == 'mlx'
    np.testing.assert_array_equal(flat['native'], np.array(native))
    restored = deserialize_tree(serialize_tree(source))
    np.testing.assert_array_equal(restored['native'], np.array(native))
    assert isinstance(restored['native'], np.ndarray)
    assert restored['none'] is None and restored['text'] == 'wire'
    assert isinstance(restored['sequence'], tuple)
    assert restored['sequence'][1].dtype == torch.bfloat16
    assert torch.equal(restored['sequence'][1], source['sequence'][1])
    before_device = str(mx.default_device())
    before_streams = [str(mx.default_stream(device)) for device in (mx.cpu, mx.gpu)]
    before_rng = [np.array(value) for value in mx.random.state]
    with configure_environment(configuration(cpu=True, model_seed=37)):
        model = models.get()
        assert type(model) is LeNet5
        assert models.registered_mlx_models['mlx_lenet5'] is Model
        trainer = ComposableMLXTrainer(model=model)
        algorithm = algorithms.get(trainer)
        assert type(algorithm) is Algorithm
        assert algorithms.registered_algorithms['mlx_fedavg'] is Algorithm
        try:
            algorithms.get()
        except TypeError as exc:
            assert 'native MLX trainer' in str(exc)
        else:
            raise AssertionError('native trainer contract must remain strict')
    assert before_device == str(mx.default_device())
    assert before_streams == [str(mx.default_stream(device)) for device in (mx.cpu, mx.gpu)]
    for before, after in zip(before_rng, mx.random.state, strict=True):
        np.testing.assert_array_equal(before, np.array(after))
    record.update({'tree_loaded_before_mlx_detects_native_array': True, 'mixed_tree_serialization_preserved': True, 'registry_constructors_resolve_to_exact_native_objects': True, 'device_streams_rng_preserved': True, 'real_native_trainer_type_guard_preserved': True})
output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
