"""
FedADP aggregation strategy delegating framework-specific ops to the Algorithm.

Reference:

H. Wu, P. Wang. "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Trans. on Cognitive Communications and Networking (TCCN), 2021.
"""

from types import SimpleNamespace

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedADPAggregationStrategy(AggregationStrategy):
    """FedAdp aggregation driven by algorithm-implemented tensor ops."""

    def __init__(self, alpha: float = 5):
        super().__init__()
        self.alpha = alpha

    def setup(self, context: ServerContext) -> None:
        """Initialize alpha from the configuration if provided."""
        if hasattr(Config().algorithm, "alpha"):
            self.alpha = Config().algorithm.alpha

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        """Delegate FedAdp aggregation to the Algorithm."""
        algorithm = getattr(context, "algorithm", None)
        if algorithm is None:
            raise RuntimeError("FedAdp requires an algorithm instance in context.")

        alpha = (
            Config().algorithm.alpha
            if hasattr(Config().algorithm, "alpha")
            else self.alpha
        )

        return algorithm.fedadp_aggregate_deltas(
            updates, deltas_received, alpha=alpha, current_round=context.current_round
        )
