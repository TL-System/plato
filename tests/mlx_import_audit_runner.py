"""Audit full production imports in a fresh process, without replacing hooks."""

import importlib.abc
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pytest


def main() -> int:
    """Run a bounded caller-selected collection and retain backend import order."""
    output, mode, *arguments = sys.argv[1:]
    lookups = []

    class Audit(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "mlx" or fullname.startswith("mlx."):
                frame = sys._getframe(1)
                preflight = False
                upstream_presence = False
                while frame is not None:
                    if (
                        frame.f_code.co_name == "_native_prerequisites"
                        and Path(frame.f_code.co_filename).name == "conftest.py"
                    ):
                        preflight = True
                    if (
                        fullname == "mlx"
                        and frame.f_code.co_name == "_is_package_available"
                        and frame.f_locals.get("return_version") is False
                        and Path(frame.f_code.co_filename).parts[-3:]
                        == ("transformers", "utils", "import_utils.py")
                    ):
                        upstream_presence = True
                    frame = frame.f_back
                lookups.append(
                    {
                        "module": fullname,
                        "inside_preflight": preflight,
                        "upstream_presence_query": upstream_presence,
                    }
                )
                if mode == "block" and not upstream_presence:
                    raise RuntimeError(
                        "MLX import sentinel reached outside native scope"
                    )
            return None

    try:
        version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        version = None
    sys.meta_path.insert(0, Audit())
    code = int(pytest.main([*arguments, "-p", "no:cacheprovider", "-q"]))
    modules = list(sys.modules)
    record = {
        "exit_status": code,
        "mlx_distribution_version": version,
        "platform": [platform.system(), platform.machine()],
        "backend_spec_lookups": lookups,
        "backend_modules": [
            name for name in modules if name == "mlx" or name.startswith("mlx.")
        ],
        "native_test_modules": [
            name
            for name in modules
            if ".test_phase3_" in name or name.startswith("test_phase3_")
        ],
        "native_frontdoor_modules": [
            name
            for name in modules
            if name
            in {
                "plato.trainers.mlx",
                "plato.algorithms.mlx_fedavg",
                "plato.models.mlx.lenet5",
            }
        ],
        "production_conftest_loaded": "tests.conftest" in modules,
    }
    Path(output).write_text(json.dumps(record, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
