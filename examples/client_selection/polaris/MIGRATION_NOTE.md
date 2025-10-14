# Polaris Migration to Strategy Pattern

## Current Implementation

The Polaris server (182 lines) implements both:
1. **Custom Aggregation** (`aggregate_deltas`) for tracking gradient bounds
2. **Custom Selection** (`choose_clients`) with probability-based selection using geometric optimization

## Migration Requirements

Polaris would require TWO strategies:

### 1. PolarisAggregationStrategy

```python
class PolarisAggregationStrategy(AggregationStrategy):
    """
    Polaris aggregation with gradient bound tracking.

    Tracks squared deltas (gradient norms) for each client and
    estimates bounds for unexplored clients based on explored averages.
    """

    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha
        self.squared_deltas_current_round = None
        self.unexplored_clients = None

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext
    ) -> Dict:
        """Aggregate and track gradient bounds."""
        # 1. Call parent FedAvg aggregation
        # 2. Calculate squared delta (L2 norm) for each client
        # 3. Remove explored clients from unexplored list
        # 4. Estimate expected deltas for unexplored clients
        ...
```

### 2. PolarisSelectionStrategy

```python
class PolarisSelectionStrategy(ClientSelectionStrategy):
    """
    Polaris client selection using geometric programming optimization.

    Solves a convex optimization problem to find optimal sampling probabilities
    that balance:
    - Aggregation weight variance (minimize)
    - Staleness × gradient bound (minimize)

    Uses MOSEK solver via cvxopt.
    """

    def __init__(self):
        super().__init__()
        self.local_gradient_bounds = None
        self.aggregation_weights = None
        self.local_stalenesses = None

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext
    ) -> List[int]:
        """Select clients based on optimized probability distribution."""
        # 1. Calculate selection probability via geometric programming
        # 2. Sample clients according to calculated probabilities
        ...

    def calculate_selection_probability(self, clients_pool):
        """
        Solve geometric programming problem:

        Minimize β Σ(p_i² G_i² / q_i) + A Σ(q_i τ_i G_i)
        Subject to: Σq_i = 1, q_i > 0

        Where:
        - q_i: selection probability (variable)
        - p_i: aggregation weight
        - G_i: gradient bound
        - τ_i: staleness

        Uses cvxopt.solvers.gp with MOSEK backend.
        """
        ...

    def on_reports_received(
        self,
        updates: List[SimpleNamespace],
        context: ServerContext
    ) -> None:
        """Update staleness, gradient bounds, and aggregation weights."""
        # 1. Extract staleness from updates
        # 2. Update local gradient bounds from tracked deltas
        # 3. Calculate aggregation weights (sample proportions)
        ...
```

### 3. Migrated Server

```python
from plato.servers import fedavg

class Server(fedavg.Server):
    """Polaris server using strategy pattern."""

    def __init__(self, model=None, datasource=None, algorithm=None,
                 trainer=None, callbacks=None):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=PolarisAggregationStrategy(alpha=10),
            client_selection_strategy=PolarisSelectionStrategy()
        )
```

## Dependencies

Polaris requires:
- `cvxopt` for geometric programming solver
- `mosek` as the optimization backend
- `numpy` for numerical computations

## Complexity

Polaris is complex because it:
1. Combines aggregation and selection (needs both strategies)
2. Uses advanced optimization (geometric programming)
3. Requires external solver libraries (cvxopt, MOSEK)
4. Tracks gradient bounds and staleness
5. Coordinates state between strategies

Total implementation: ~250 lines for both strategies

## Mathematical Formulation

The selection probability optimization minimizes a combination of:
- **Variance term**: β Σ(p_i² G_i² / q_i) - reduces aggregation variance
- **Staleness term**: A Σ(q_i τ_i G_i) - reduces staleness impact

This is a convex problem solvable via geometric programming.

## Current Workaround

Continue using the inheritance-based approach:

```python
from polaris_server import Server

server = Server()
server.run(client)
```

## Future Work

This would be an advanced addition to the strategy library, showcasing:
- Integration with optimization solvers
- Probability-based client selection
- Coordination between aggregation and selection strategies
- Mathematical programming in FL
