import ast
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

source_path = Path('/tmp/plato-refresh-worktrees/mlx-profile/tests/conftest.py')
source = source_path.read_text()
node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == '_native_prerequisites')
namespace = {'platform': platform, 'importlib': importlib, 'math': math, 'pytest': pytest}
exec(compile(ast.get_source_segment(source, node), str(source_path), 'exec'), namespace)
before_device = str(mx.default_device())
before_streams = [str(mx.default_stream(device)) for device in (mx.cpu, mx.gpu)]
before_rng = [np.array(value) for value in mx.random.state]
start = time.monotonic()
namespace['_native_prerequisites']()
elapsed = time.monotonic() - start
assert before_device == str(mx.default_device())
assert before_streams == [str(mx.default_stream(device)) for device in (mx.cpu, mx.gpu)]
for before, after in zip(before_rng, mx.random.state, strict=True):
    np.testing.assert_array_equal(before, np.array(after))
receipt = {'schema_version': 1, 'kind': 'actual production prerequisite probe; not E1-E7 qualification', 'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'python': sys.version, 'platform': platform.platform(), 'versions': {name: importlib.metadata.version(name) for name in ('mlx', 'mlx-metal', 'pytest')}, 'elapsed_seconds': elapsed, 'default_device': before_device, 'default_streams': before_streams, 'caller_device_streams_rng_restored': True}
Path('/tmp/plato-mlx-profile-real-preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
