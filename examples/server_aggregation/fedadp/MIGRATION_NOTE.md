# FedADP Migration to Strategy Pattern

## Current Implementation

The FedADP server (140 lines) overrides `aggregate_deltas()` with custom adaptive weighting logic based on:
- Gradient angle calculations between local and global gradients
- Smoothed angle tracking per client
- Non-linear contribution mapping
- Exponential weighting based on contribution and data size

## Migration Requirements

To fully migrate FedADP to the strategy pattern, you would need to:

### 1. Create FedADPAggregationStrategy

```python
class FedADPAggregationStrategy(AggregationStrategy):
    """
    FedADP aggregation with adaptive weighting.

    Uses gradient angles to compute client contributions and
    applies exponential weighting for aggregation.
    """

    def __init__(self, alpha=5):
        super().__init__()
        self.alpha = alpha
        self.local_angles = {}
        self.last_global_grads = None

    def setup(self, context: ServerContext):
        """Initialize from config if needed."""
        if hasattr(Config().algorithm, "alpha"):
            self.alpha = Config().algorithm.alpha

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext
    ) -> Dict:
        """Aggregate using adaptive weighting based on gradient angles."""
        # 1. Calculate global gradients (weighted average)
        # 2. Calculate local gradient angles
        # 3. Update smoothed angles per client
        # 4. Calculate client contributions
        # 5. Compute adaptive weights (contribution + data size)
        # 6. Perform weighted aggregation
        ...
```

### 2. Key Components to Extract

- `calc_adaptive_weighting()` → part of strategy's aggregate logic
- `calc_contribution()` → private method in strategy
- `process_grad()` → static utility method in strategy
- State: `self.local_angles`, `self.last_global_grads`, `self.global_grads`

### 3. Migrated Server

```python
from plato.servers import fedavg

class Server(fedavg.Server):
    """FedADP server using strategy pattern."""

    def __init__(self, model=None, datasource=None, algorithm=None,
                 trainer=None, callbacks=None):
        # Load alpha from config
        alpha = Config().algorithm.alpha if hasattr(Config().algorithm, "alpha") else 5

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=FedADPAggregationStrategy(alpha=alpha)
        )
```

## Why This Wasn't Implemented

FedADP has complex state management and gradient processing logic that would require:
1. A full strategy class implementation (~150 lines)
2. Careful handling of gradient conversions (model weights → numpy arrays)
3. State persistence across rounds (local_angles dictionary)
4. Integration with the trainer's tensor operations

This is a perfect candidate for a future contribution to the strategy library!

## Current Workaround

Continue using the inheritance-based approach:

```python
from fedadp_server import Server

server = Server()
server.run(client)
```

The backward compatibility ensures this will continue to work.
