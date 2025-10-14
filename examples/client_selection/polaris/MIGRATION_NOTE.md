# Polaris Migration to Strategy Pattern

Polaris now uses the shared server-strategy API instead of subclassing `fedavg.Server`.

## What Changed

- Added `PolarisAggregationStrategy` and `PolarisSelectionStrategy` to
  `plato.servers.strategies`, encapsulating the original gradient-bound tracking
  and geometric-programming client selection.
- `polaris_server.Server` became a thin wrapper that wires those strategies into
  `fedavg.Server`, reading any Polaris-specific parameters from `Config().server`.
- All example entry points (`polaris.py`) continue to function without code changes.

## Key Strategy Details

- `PolarisAggregationStrategy` extends FedAvg aggregation while:
  - Computing convolutional gradient norms for reporting clients.
  - Estimating bounds for unseen clients via `alpha * avg_delta`.
  - Persisting per-client squared deltas for the selection strategy.
- `PolarisSelectionStrategy` solves the original geometric program:
  - Minimizes β Σ(p_i² G_i² / q_i) + A Σ(q_i τ_i G_i) subject to Σq_i = 1.
  - Requires `cvxopt` with the MOSEK backend (`pip install cvxopt`, MOSEK license).
  - Updates per-client staleness and aggregation weights from incoming reports before
    sampling the next cohort.

## Usage

```python
from plato.servers import fedavg
from plato.servers.strategies import (
    PolarisAggregationStrategy,
    PolarisSelectionStrategy,
)

server = fedavg.Server(
    aggregation_strategy=PolarisAggregationStrategy(alpha=10.0),
    client_selection_strategy=PolarisSelectionStrategy(beta=1.0, staleness_weight=1.0),
)
```

The example `polaris_server.Server` already applies this wiring, so scripts can keep
importing `polaris_server` as before. Ensure the required optimization dependencies
(`cvxopt`, `mosek`) are installed when running Polaris.
