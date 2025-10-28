"""
Registry for benchmarks.

Enables runtime benchmark selection via configuration.
"""

from plato.benchmarks import core
from plato.benchmarks.base import Benchmark as BenchmarkBase

registered_benchmarks: dict[str, type[BenchmarkBase]] = {
    "core": core.Benchmark,
}


def get(type: str) -> BenchmarkBase:
    """Get an instance of the benchmark."""
    if type in registered_benchmarks:
        benchmark_cls = registered_benchmarks[type]
        registered_benchmark = benchmark_cls()
    else:
        available = list(registered_benchmarks.keys())
        raise ValueError(
            f"No such benchmark: {type}. Available benchmarks: {available}"
        )

    return registered_benchmark


def list_benchmarks():
    """List all available benchmark types."""
    return list(registered_benchmarks.keys())
