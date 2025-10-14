# Pisces Migration to Strategy Pattern

Pisces now uses the shared strategy API rather than subclassing `fedavg.Server`.

## What Changed

- Added `PiscesAggregationStrategy` and `PiscesSelectionStrategy` under
  `plato.servers.strategies`. They implement the original staleness-aware
  aggregation, exploration/exploitation policy, and optional robustness
  (DBSCAN-based anomaly detection with reliability credits).
- `pisces_server.Server` is now a thin wrapper that wires these strategies into
  `fedavg.Server`, loading all parameters from `Config().server`.
- All example entry points (`pisces.py`) continue to work without changes.

## Key Strategy Details

- `PiscesAggregationStrategy` keeps a sliding history of client staleness and
  applies the Pisces decay factor:
  `factor = 1 / (staleness + 1) ** staleness_factor`.
- `PiscesSelectionStrategy` maintains exploration vs. exploitation with a
  decaying exploration rate, staleness-adjusted utilities, and optional
  robustness:
  - Pools recent utilities across model versions (up to `augmented_factor`)
    and runs DBSCAN.
    Clients whose reliability credit reaches zero are filtered out in future
    rounds.
  - Configurable parameters mirror the legacy implementation
    (`exploration_factor`, `exploration_decaying_factor`,
    `min_explore_factor`, `staleness_factor`, `augmented_factor`,
    `threshold_factor`, `reliability_credit_initial`, `robustness`).

## Usage

```python
from plato.servers import fedavg
from plato.servers.strategies import (
    PiscesAggregationStrategy,
    PiscesSelectionStrategy,
)

server = fedavg.Server(
    aggregation_strategy=PiscesAggregationStrategy(),
    client_selection_strategy=PiscesSelectionStrategy(),
)
```

The example `pisces_server.Server` already applies this wiring, so existing
scripts can keep importing `pisces_server` as before.
