# Pisces Migration to Strategy Pattern

## Current Implementation

The Pisces server (246 lines) implements both:
1. **Custom Aggregation** (`aggregate_deltas`) with staleness-aware weighting
2. **Custom Selection** (`choose_clients`) with exploration/exploitation and optional robustness (outlier detection)

## Migration Requirements

Pisces would require TWO strategies:

### 1. PiscesAggregationStrategy

```python
class PiscesAggregationStrategy(AggregationStrategy):
    """
    Pisces aggregation with staleness-aware weighting.

    Applies staleness factor to downweight stale updates:
    factor = 1.0 / (staleness + 1)^staleness_factor
    """

    def __init__(self, staleness_factor=1.0):
        super().__init__()
        self.staleness_factor = staleness_factor
        self.client_staleness = {}

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext
    ) -> Dict:
        """Aggregate with staleness weighting."""
        # 1. Track client staleness history
        # 2. Calculate staleness factor per client
        # 3. Apply weighted averaging with staleness
        ...
```

### 2. PiscesSelectionStrategy

```python
class PiscesSelectionStrategy(ClientSelectionStrategy):
    """
    Pisces client selection with utility-based selection and optional robustness.

    Combines:
    - Statistical utility
    - Staleness factor
    - Exploration/exploitation with decaying exploration
    - Optional outlier detection via DBSCAN
    """

    def __init__(self, exploration_factor=0.3, exploration_decaying_factor=0.99,
                 min_explore_factor=0.1, robustness=False):
        super().__init__()
        self.exploration_factor = exploration_factor
        self.exploration_decaying_factor = exploration_decaying_factor
        self.min_explore_factor = min_explore_factor
        self.robustness = robustness

        # State
        self.client_utilities = {}
        self.explored_clients = []
        self.unexplored_clients = []
        self.reliability_credit_record = {}
        self.detected_corrupted_clients = []

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext
    ) -> List[int]:
        """Select clients using utility-based exploration/exploitation."""
        # 1. Filter out detected outliers (if robustness enabled)
        # 2. Exploitation: select high-utility explored clients
        # 3. Exploration: random sample from unexplored clients
        # 4. Decay exploration factor
        ...

    def on_reports_received(
        self,
        updates: List[SimpleNamespace],
        context: ServerContext
    ) -> None:
        """Update utilities and detect outliers if enabled."""
        # 1. Calculate client utilities (stat_utility * staleness_factor)
        # 2. Pool updates by model version (if robustness enabled)
        # 3. Run DBSCAN outlier detection (if robustness enabled)
        # 4. Update reliability credits
        ...
```

### 3. Migrated Server

```python
from plato.servers import fedavg

class Server(fedavg.Server):
    """Pisces server using strategy pattern."""

    def __init__(self, model=None, datasource=None, algorithm=None,
                 trainer=None, callbacks=None):
        # Load parameters from config
        staleness_factor = Config().server.staleness_factor
        exploration_factor = Config().server.exploration_factor
        exploration_decaying_factor = Config().server.exploration_decaying_factor
        min_explore_factor = Config().server.min_explore_factor

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=PiscesAggregationStrategy(
                staleness_factor=staleness_factor
            ),
            client_selection_strategy=PiscesSelectionStrategy(
                exploration_factor=exploration_factor,
                exploration_decaying_factor=exploration_decaying_factor,
                min_explore_factor=min_explore_factor,
                robustness=False  # or load from config
            )
        )
```

## Complexity

Pisces is complex because it:
1. Combines aggregation and selection (needs both strategies)
2. Has outlier detection with DBSCAN clustering
3. Maintains multiple state dictionaries
4. Tracks model version history for robustness

Total implementation: ~300 lines for both strategies

## Current Workaround

Continue using the inheritance-based approach:

```python
from pisces_server import Server

server = Server()
server.run(client)
```

## Future Work

This would be an excellent addition to the strategy library, showcasing:
- Combined aggregation + selection strategies
- Robustness features (outlier detection)
- Decaying exploration patterns
