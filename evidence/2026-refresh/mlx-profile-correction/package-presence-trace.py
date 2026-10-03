import importlib.abc
import sys
import traceback
import pytest
class Trace(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            print('MLX LOOKUP STACK', fullname, flush=True)
            traceback.print_stack(limit=24)
        return None
sys.meta_path.insert(0, Trace())
raise SystemExit(pytest.main(['--pyargs', 'tests.mlx_native.test_phase3_controls', '-q', '-p', 'no:cacheprovider']))
