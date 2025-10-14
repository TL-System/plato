Strategy Pattern Migration Plan for Plato Server API

  Executive Summary

  This plan introduces composition-based strategy patterns for aggregation and client selection in Plato's server API, while preserving the existing inheritance-based lifecycle hooks. This hybrid approach
  provides:

  - ✅ Better composability and testability for core algorithms
  - ✅ Backward compatibility with existing examples
  - ✅ Gradual migration path
  - ✅ Consistency with the composable trainer architecture

  ---
  Phase 1: Foundation - Strategy Interfaces & Context

  1.1 Create Server Strategy Base Module

  File: `plato/servers/strategies/base.py`

```python
  """
  Base strategy interfaces for composable server architecture.

  Similar to plato.trainers.strategies.base, this module defines strategy
  interfaces for server-side federated learning operations.
  """

  from abc import ABC, abstractmethod
  from typing import Any, Dict, List, Optional
  from types import SimpleNamespace


  class ServerContext:
      """
      Shared context passed between server strategies during FL training.

      Attributes:
          server: Reference to the server instance
          trainer: The server's trainer instance
          algorithm: The aggregation algorithm instance
          current_round: Current FL round number
          total_clients: Total number of clients
          clients_per_round: Number of clients selected per round
          updates: List of client updates in current round
          state: Dictionary for strategies to share arbitrary state
      """

      def __init__(self):
          self.server = None  # Reference to server
          self.trainer = None
          self.algorithm = None
          self.current_round: int = 0
          self.total_clients: int = 0
          self.clients_per_round: int = 0
          self.updates: List[SimpleNamespace] = []
          self.state: Dict[str, Any] = {}


  class ServerStrategy(ABC):
      """Base class for all server strategies."""

      def setup(self, context: ServerContext) -> None:
          """Called once during server initialization."""
          pass

      def teardown(self, context: ServerContext) -> None:
          """Called when server closes."""
          pass


  class AggregationStrategy(ServerStrategy):
      """
      Strategy interface for aggregating client model updates.

      Implement this to customize:
      - FedAvg weighted averaging
      - FedNova normalized aggregation
      - FedAsync staleness-aware aggregation
      - Custom aggregation algorithms
      """

      @abstractmethod
      async def aggregate_deltas(
          self,
          updates: List[SimpleNamespace],
          deltas_received: List[Dict],
          context: ServerContext
      ) -> Dict:
          """
          Aggregate weight deltas from clients.

          Args:
              updates: List of client update objects with metadata
              deltas_received: List of weight delta dictionaries
              context: Server context with state and references

          Returns:
              Aggregated weight deltas as dictionary
          """
          pass

      def aggregate_weights(
          self,
          updates: List[SimpleNamespace],
          baseline_weights: Dict,
          weights_received: List[Dict],
          context: ServerContext
      ) -> Optional[Dict]:
          """
          Optional: Aggregate weights directly instead of deltas.

          Args:
              updates: List of client update objects
              baseline_weights: Current global model weights
              weights_received: List of client weight dictionaries
              context: Server context

          Returns:
              Aggregated weights, or None to use aggregate_deltas instead
          """
          return None  # Default: use delta aggregation


  class ClientSelectionStrategy(ServerStrategy):
      """
      Strategy interface for selecting clients each round.

      Implement this to customize:
      - Random selection (default)
      - Oort utility-based selection
      - AFL (Active Federated Learning)
      - Power-of-choice
      - Custom selection algorithms
      """

      @abstractmethod
      def select_clients(
          self,
          clients_pool: List[int],
          clients_count: int,
          context: ServerContext
      ) -> List[int]:
          """
          Select a subset of clients for the current round.

          Args:
              clients_pool: List of available client IDs
              clients_count: Number of clients to select
              context: Server context with round info and state

          Returns:
              List of selected client IDs
          """
          pass

      def on_clients_selected(
          self,
          selected_clients: List[int],
          context: ServerContext
      ) -> None:
          """Hook called after clients are selected."""
          pass

      def on_reports_received(
          self,
          updates: List[SimpleNamespace],
          context: ServerContext
      ) -> None:
          """Hook called after client reports are received."""
          pass
```

  1.2 Create Default Strategy Implementations

```python
  File: plato/servers/strategies/aggregation.py

  """Default aggregation strategy implementations."""

  import asyncio
  from typing import Dict, List
  from types import SimpleNamespace

  from plato.servers.strategies.base import AggregationStrategy, ServerContext


  class FedAvgAggregationStrategy(AggregationStrategy):
      """
      Standard Federated Averaging aggregation.

      Performs weighted averaging of client deltas based on number of samples.
      """

      async def aggregate_deltas(
          self,
          updates: List[SimpleNamespace],
          deltas_received: List[Dict],
          context: ServerContext
      ) -> Dict:
          """Aggregate using weighted average by sample count."""
          # Extract total number of samples
          total_samples = sum(update.report.num_samples for update in updates)

          # Initialize aggregated deltas
          avg_update = {
              name: context.trainer.zeros(delta.shape)
              for name, delta in deltas_received[0].items()
          }

          # Weighted averaging
          for i, update in enumerate(deltas_received):
              num_samples = updates[i].report.num_samples

              for name, delta in update.items():
                  avg_update[name] += delta * (num_samples / total_samples)

              # Yield to other async tasks
              await asyncio.sleep(0)

          return avg_update


  class FedNovaAggregationStrategy(AggregationStrategy):
      """
      FedNova aggregation with normalized momentum.

      Reference:
      Wang et al., "Tackling the Objective Inconsistency Problem in
      Heterogeneous Federated Optimization", NeurIPS 2020.
      """

      async def aggregate_deltas(
          self,
          updates: List[SimpleNamespace],
          deltas_received: List[Dict],
          context: ServerContext
      ) -> Dict:
          """Aggregate using FedNova normalized averaging."""
          total_samples = sum(update.report.num_samples for update in updates)
          local_epochs = [update.report.epochs for update in updates]

          avg_update = {
              name: context.trainer.zeros(delta.shape)
              for name, delta in deltas_received[0].items()
          }

          # Calculate effective tau
          tau_eff = sum(
              local_epochs[i] * updates[i].report.num_samples / total_samples
              for i in range(len(updates))
          )

          # Normalized aggregation
          for i, update in enumerate(deltas_received):
              num_samples = updates[i].report.num_samples

              for name, delta in update.items():
                  avg_update[name] += (
                      delta
                      * (num_samples / total_samples)
                      * tau_eff
                      / local_epochs[i]
                  )

          return avg_update


  class FedAsyncAggregationStrategy(AggregationStrategy):
      """
      FedAsync aggregation with staleness-aware mixing.

      Reference:
      Xie et al., "Asynchronous federated optimization", OPT 2020.
      """

      def __init__(self, mixing_hyperparameter: float = 0.9, adaptive: bool = False):
          self.mixing_hyperparam = mixing_hyperparameter
          self.adaptive_mixing = adaptive

      async def aggregate_weights(
          self,
          updates: List[SimpleNamespace],
          baseline_weights: Dict,
          weights_received: List[Dict],
          context: ServerContext
      ) -> Dict:
          """Aggregate weights directly with staleness mixing."""
          # Adjust mixing based on staleness if adaptive
          client_staleness = updates[0].staleness
          mixing = self.mixing_hyperparam

          if self.adaptive_mixing:
              mixing *= self._staleness_function(client_staleness)

          # Use algorithm's aggregate_weights with mixing parameter
          return await context.algorithm.aggregate_weights(
              baseline_weights, weights_received, mixing=mixing
          )

      @staticmethod
      def _staleness_function(staleness: int) -> float:
          """Simple polynomial staleness function."""
          return (staleness + 1) ** -1.0
```
  File: `plato/servers/strategies/client_selection.py`

```python
  """Client selection strategy implementations."""

  import logging
  import math
  import random
  from typing import List

  import numpy as np

  from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext
  from types import SimpleNamespace


  class RandomSelectionStrategy(ClientSelectionStrategy):
      """
      Random client selection (default strategy).

      Selects clients uniformly at random from the pool.
      """

      def select_clients(
          self,
          clients_pool: List[int],
          clients_count: int,
          context: ServerContext
      ) -> List[int]:
          """Select clients uniformly at random."""
          assert clients_count <= len(clients_pool)

          # Use server's PRNG state for reproducibility
          prng_state = context.state.get('prng_state')
          if prng_state:
              random.setstate(prng_state)

          selected_clients = random.sample(clients_pool, clients_count)

          # Save PRNG state back
          context.state['prng_state'] = random.getstate()

          logging.info("[Server] Selected clients: %s", selected_clients)
          return selected_clients


  class OortSelectionStrategy(ClientSelectionStrategy):
      """
      Oort utility-based client selection.

      Combines exploration and exploitation using client utilities,
      training times, and staleness.

      Reference:
      Lai et al., "Oort: Efficient Federated Learning via Guided
      Participant Selection", OSDI 2021.
      """

      def __init__(
          self,
          exploration_factor: float = 0.3,
          desired_duration: float = 100.0,
          step_window: int = 10,
          penalty: float = 0.8,
          cut_off: float = 0.95,
          blacklist_num: int = 10
      ):
          self.exploration_factor = exploration_factor
          self.desired_duration = desired_duration
          self.step_window = step_window
          self.penalty = penalty
          self.cut_off = cut_off
          self.blacklist_num = blacklist_num

          # State maintained across rounds
          self.blacklist = []
          self.client_utilities = {}
          self.client_durations = {}
          self.client_last_rounds = {}
          self.client_selected_times = {}
          self.explored_clients = []
          self.unexplored_clients = []
          self.util_history = []
          self.pacer_step = desired_duration

      def setup(self, context: ServerContext) -> None:
          """Initialize client tracking dictionaries."""
          total_clients = context.total_clients

          self.client_utilities = {
              client_id: 0 for client_id in range(1, total_clients + 1)
          }
          self.client_durations = {
              client_id: 0 for client_id in range(1, total_clients + 1)
          }
          self.client_last_rounds = {
              client_id: 0 for client_id in range(1, total_clients + 1)
          }
          self.client_selected_times = {
              client_id: 0 for client_id in range(1, total_clients + 1)
          }
          self.unexplored_clients = list(range(1, total_clients + 1))

      def select_clients(
          self,
          clients_pool: List[int],
          clients_count: int,
          context: ServerContext
      ) -> List[int]:
          """Select clients using Oort algorithm."""
          selected_clients = []
          current_round = context.current_round

          if current_round > 1:
              # Exploitation phase
              exploited_clients_count = max(
                  math.ceil((1.0 - self.exploration_factor) * clients_count),
                  clients_count - len(self.unexplored_clients)
              )

              sorted_by_utility = sorted(
                  self.client_utilities,
                  key=self.client_utilities.get,
                  reverse=True
              )
              sorted_by_utility = [
                  client for client in sorted_by_utility
                  if client in clients_pool
              ]

              # Calculate cut-off utility
              if len(sorted_by_utility) >= exploited_clients_count:
                  cut_off_util = (
                      self.client_utilities[
                          sorted_by_utility[exploited_clients_count - 1]
                      ] * self.cut_off
                  )
              else:
                  cut_off_util = 0

              # Select high-utility clients
              exploited_clients = [
                  client_id for client_id in sorted_by_utility
                  if (self.client_utilities[client_id] > cut_off_util
                      and client_id not in self.blacklist)
              ]

              # Sample with utility-based probabilities
              if exploited_clients:
                  total_utility = float(sum(
                      self.client_utilities[cid] for cid in exploited_clients
                  ))

                  if total_utility > 0:
                      probabilities = np.array([
                          self.client_utilities[cid] / total_utility
                          for cid in exploited_clients
                      ])
                      probabilities = probabilities / probabilities.sum()

                      selected_clients = np.random.choice(
                          exploited_clients,
                          min(len(exploited_clients), exploited_clients_count),
                          p=probabilities,
                          replace=False
                      ).tolist()

          # Exploration phase
          prng_state = context.state.get('prng_state')
          if prng_state:
              random.setstate(prng_state)

          remaining_count = clients_count - len(selected_clients)
          if remaining_count > 0 and self.unexplored_clients:
              selected_unexplore = random.sample(
                  self.unexplored_clients,
                  min(remaining_count, len(self.unexplored_clients))
              )

              self.explored_clients += selected_unexplore
              for client_id in selected_unexplore:
                  self.unexplored_clients.remove(client_id)

              selected_clients += selected_unexplore

          context.state['prng_state'] = random.getstate()

          # Track selection counts
          for client in selected_clients:
              self.client_selected_times[client] += 1

          logging.info("[Server] Selected clients: %s", selected_clients)
          return selected_clients

      def on_reports_received(
          self,
          updates: List[SimpleNamespace],
          context: ServerContext
      ) -> None:
          """Update client utilities and durations after reports."""
          for update in updates:
              client_id = update.client_id

              # Update utilities and durations
              self.client_utilities[client_id] = update.report.statistical_utility
              self.client_durations[client_id] = update.report.training_time
              self.client_last_rounds[client_id] = context.current_round

              # Recalculate utility
              self.client_utilities[client_id] = self._calc_client_util(
                  client_id, context.current_round
              )

          # Adjust pacer
          self.util_history.append(
              sum(update.report.statistical_utility for update in updates)
          )

          if context.current_round >= 2 * self.step_window:
              last_pacer = sum(
                  self.util_history[-2 * self.step_window: -self.step_window]
              )
              current_pacer = sum(self.util_history[-self.step_window:])
              if last_pacer > current_pacer:
                  self.desired_duration += self.pacer_step

          # Blacklist clients exceeding selection threshold
          for update in updates:
              if self.client_selected_times[update.client_id] > self.blacklist_num:
                  if update.client_id not in self.blacklist:
                      self.blacklist.append(update.client_id)

      def _calc_client_util(self, client_id: int, current_round: int) -> float:
          """Calculate client utility with exploration bonus and penalty."""
          client_utility = self.client_utilities[client_id] + math.sqrt(
              0.1 * math.log(current_round) / max(1, self.client_last_rounds[client_id])
          )

          # Apply duration penalty if client is too slow
          if self.desired_duration < self.client_durations[client_id]:
              global_utility = (
                  self.desired_duration / self.client_durations[client_id]
              ) ** self.penalty
              client_utility *= global_utility

          return client_utility


  class AFLSelectionStrategy(ClientSelectionStrategy):
      """
      Active Federated Learning (AFL) client selection.

      Selects clients based on valuation, which measures how much
      a client can improve the global model.

      Reference:
      Goetz et al., "Active Federated Learning", 2019.
      """

      def __init__(self, alpha1: float = 0.75, alpha2: float = 0.01, alpha3: float = 0.1):
          self.alpha1 = alpha1  # Proportion to reset valuations
          self.alpha2 = alpha2  # Temperature for sampling
          self.alpha3 = alpha3  # Proportion for uniform sampling
          self.local_values = {}

      def select_clients(
          self,
          clients_pool: List[int],
          clients_count: int,
          context: ServerContext
      ) -> List[int]:
          """Select clients using AFL algorithm."""
          # Initialize new clients
          for client_id in clients_pool:
              if client_id not in self.local_values:
                  self.local_values[client_id] = {
                      'valuation': -float('inf'),
                      'prob': 0.0
                  }

          # Update sampling distribution
          self._calc_sample_distribution(clients_pool)

          prng_state = context.state.get('prng_state')
          if prng_state:
              random.setstate(prng_state)

          # Phase 1: Sample based on valuations
          num1 = int(math.floor((1 - self.alpha3) * clients_count))
          probs = np.array([
              self.local_values[cid]['prob'] for cid in clients_pool
          ])

          # Add small probability to zeros
          probs = probs + 0.01
          probs /= probs.sum()

          subset1 = np.random.choice(
              clients_pool, num1, p=probs, replace=False
          ).tolist()

          # Phase 2: Uniform random sampling
          num2 = clients_count - num1
          remaining = [c for c in clients_pool if c not in subset1]
          subset2 = random.sample(remaining, num2)

          selected_clients = subset1 + subset2

          context.state['prng_state'] = random.getstate()

          logging.info("[Server] Selected clients: %s", selected_clients)
          return selected_clients

      def on_reports_received(
          self,
          updates: List[SimpleNamespace],
          context: ServerContext
      ) -> None:
          """Extract valuations from client reports."""
          for update in updates:
              if hasattr(update.report, 'valuation'):
                  self.local_values[update.client_id]['valuation'] = (
                      update.report.valuation
                  )

      def _calc_sample_distribution(self, clients_pool: List[int]) -> None:
          """Calculate sampling probabilities for clients."""
          # Reset smallest valuations
          num_smallest = int(self.alpha1 * len(clients_pool))
          sorted_clients = sorted(
              self.local_values.items(),
              key=lambda x: x[1]['valuation']
          )[:num_smallest]

          for client_id, _ in sorted_clients:
              self.local_values[client_id]['valuation'] = -float('inf')

          # Calculate probabilities
          for client_id in clients_pool:
              self.local_values[client_id]['prob'] = math.exp(
                  self.alpha2 * self.local_values[client_id]['valuation']
              )
```
  ---
  Phase 2: Server Integration

  2.1 Modify Base Server Class

  File: `plato/servers/base.py (modifications)`

```python
  # Add imports at top
  from plato.servers.strategies.base import ServerContext
  from plato.servers.strategies.client_selection import RandomSelectionStrategy

  class Server:
      """The base class for federated learning servers."""

      def __init__(
          self,
          callbacks=None,
          client_selection_strategy=None  # NEW
      ):
          # ... existing initialization ...

          # NEW: Initialize server context
          self.context = ServerContext()

          # NEW: Initialize client selection strategy
          self.client_selection_strategy = (
              client_selection_strategy or RandomSelectionStrategy()
          )

          # ... rest of existing code ...

      def configure(self) -> None:
          """Initializes configuration settings."""
          # ... existing code ...

          # NEW: Setup server context
          self.context.server = self
          self.context.total_clients = self.total_clients
          self.context.clients_per_round = self.clients_per_round
          self.context.state['prng_state'] = self.prng_state

          # NEW: Setup strategies
          self.client_selection_strategy.setup(self.context)

      def choose_clients(self, clients_pool, clients_count):
          """Chooses a subset of the clients using strategy."""
          # NEW: Update context
          self.context.current_round = self.current_round
          self.context.state['prng_state'] = self.prng_state

          # NEW: Delegate to strategy
          selected_clients = self.client_selection_strategy.select_clients(
              clients_pool, clients_count, self.context
          )

          # NEW: Update PRNG state from context
          self.prng_state = self.context.state['prng_state']

          # NEW: Call strategy hook
          self.client_selection_strategy.on_clients_selected(
              selected_clients, self.context
          )

          return selected_clients
```

  2.2 Modify FedAvg Server Class

  File: `plato/servers/fedavg.py (modifications)`

```python
  # Add imports
  from plato.servers.strategies.base import ServerContext
  from plato.servers.strategies.aggregation import FedAvgAggregationStrategy
  from plato.servers.strategies.client_selection import RandomSelectionStrategy

  class Server(base.Server):
      """Federated learning server using federated averaging."""

      def __init__(
          self,
          model=None,
          datasource=None,
          algorithm=None,
          trainer=None,
          callbacks=None,
          aggregation_strategy=None,  # NEW
          client_selection_strategy=None  # NEW
      ):
          super().__init__(
              callbacks=callbacks,
              client_selection_strategy=client_selection_strategy  # NEW
          )

          # ... existing initialization ...

          # NEW: Initialize aggregation strategy
          self.aggregation_strategy = (
              aggregation_strategy or FedAvgAggregationStrategy()
          )

      def configure(self) -> None:
          """Booting the federated learning server."""
          super().configure()

          # ... existing code ...

          # NEW: Setup context for aggregation
          self.context.trainer = self.trainer
          self.context.algorithm = self.algorithm

          # NEW: Setup aggregation strategy
          self.aggregation_strategy.setup(self.context)

      async def aggregate_deltas(self, updates, deltas_received):
          """Aggregate using strategy pattern."""
          # NEW: Update context
          self.context.updates = updates
          self.context.current_round = self.current_round

          # NEW: Delegate to strategy
          avg_update = await self.aggregation_strategy.aggregate_deltas(
              updates, deltas_received, self.context
          )

          # NEW: Update total_samples (for backward compatibility)
          self.total_samples = sum(
              update.report.num_samples for update in updates
          )

          return avg_update

      async def _process_reports(self):
          """Process client reports by aggregating their weights."""
          weights_received = [update.payload for update in self.updates]

          weights_received = self.weights_received(weights_received)
          self.callback_handler.call_event("on_weights_received", self, weights_received)

          # Extract baseline weights
          baseline_weights = self.algorithm.extract_weights()

          # NEW: Update context
          self.context.updates = self.updates
          self.context.current_round = self.current_round

          # NEW: Check if strategy supports weight aggregation
          if hasattr(self, "aggregate_weights"):
              # Legacy path: subclass override
              updated_weights = await self.aggregate_weights(
                  self.updates, baseline_weights, weights_received
              )
              self.algorithm.load_weights(updated_weights)
          else:
              # NEW: Try strategy's aggregate_weights first
              updated_weights = await self.aggregation_strategy.aggregate_weights(
                  self.updates, baseline_weights, weights_received, self.context
              )

              if updated_weights is not None:
                  # Strategy provided weight aggregation
                  self.algorithm.load_weights(updated_weights)
              else:
                  # Fall back to delta aggregation
                  deltas_received = self.algorithm.compute_weight_deltas(
                      baseline_weights, weights_received
                  )
                  deltas = await self.aggregate_deltas(self.updates, deltas_received)
                  updated_weights = self.algorithm.update_weights(deltas)
                  self.algorithm.load_weights(updated_weights)

          # NEW: Notify selection strategy of reports
          self.client_selection_strategy.on_reports_received(
              self.updates, self.context
          )

          # ... rest of existing code (testing, callbacks, etc.) ...
```

  ---
  Phase 3: Example Migration

  3.1 Migration Strategy for Examples

  Approach: Provide both old and new implementations side-by-side

  Old style (still supported):
  class Server(fedavg.Server):
      def choose_clients(self, clients_pool, clients_count):
          # Custom selection logic
          return selected_clients

  New style (recommended):
  from plato.servers.strategies.client_selection import OortSelectionStrategy

  server = fedavg.Server(
      client_selection_strategy=OortSelectionStrategy(
          exploration_factor=0.3,
          desired_duration=100.0
      )
  )

  3.2 Migrate Key Examples

  Priority 1 - Complete Migration:
  1. examples/server_aggregation/fednova/ → Use FedNovaAggregationStrategy
  2. examples/async/fedasync/ → Use FedAsyncAggregationStrategy
  3. examples/client_selection/oort/ → Use OortSelectionStrategy
  4. examples/client_selection/afl/ → Use AFLSelectionStrategy

  Priority 2 - Create Hybrid Examples:
  5. examples/customized/ → Show both approaches
  6. Create examples/strategies/ folder with strategy-only examples

  3.3 Example Migration: FedNova

  Before (examples/server_aggregation/fednova/fednova_server.py):
  class Server(fedavg.Server):
      """A federated learning server using the FedNova algorithm."""

      async def aggregate_deltas(self, updates, deltas_received):
          """Aggregate weight updates using FedNova."""
          # ... 50 lines of aggregation logic ...

  After (Strategy-based):
  """
  A federated learning server using FedNova.

  This example demonstrates the new strategy-based approach.
  """

  from plato.servers import fedavg
  from plato.servers.strategies.aggregation import FedNovaAggregationStrategy


  def main():
      """Launch FedNova server using aggregation strategy."""
      server = fedavg.Server(
          aggregation_strategy=FedNovaAggregationStrategy()
      )
      server.run()


  if __name__ == "__main__":
      main()

  Config file remains the same.

  3.4 Example Migration: Oort

  Before (examples/client_selection/oort/oort_server.py):
  class Server(fedavg.Server):
      """Oort client selection."""

      def __init__(self, ...):
          super().__init__(...)
          # Initialize 8 state variables
          self.blacklist = []
          self.client_utilities = {}
          # ... etc ...

      def configure(self):
          super().configure()
          # Initialize 4 dictionaries

      def choose_clients(self, clients_pool, clients_count):
          # 100+ lines of selection logic

      def weights_aggregated(self, updates):
          # Update utilities and adjust pacer

      def calc_client_util(self, client_id):
          # Calculate utility

  After (Strategy-based):
  """
  A federated learning server using Oort.

  This example demonstrates strategy-based client selection.
  """

  from plato.config import Config
  from plato.servers import fedavg
  from plato.servers.strategies.client_selection import OortSelectionStrategy


  def main():
      """Launch Oort server using client selection strategy."""
      server = fedavg.Server(
          client_selection_strategy=OortSelectionStrategy(
              exploration_factor=Config().server.exploration_factor,
              desired_duration=Config().server.desired_duration,
              step_window=Config().server.step_window,
              penalty=Config().server.penalty,
              cut_off=getattr(Config().server, 'cut_off', 0.95),
              blacklist_num=getattr(Config().server, 'blacklist_num', 10)
          )
      )
      server.run()


  if __name__ == "__main__":
      main()

  ---
  Phase 4: Documentation & Testing

  4.1 Documentation Updates

  Create: docs/docs/references/server_strategies.md

  # Server Strategies

  ## Overview

  Plato supports composable server strategies for aggregation and client selection.
  Strategies can be mixed and matched or customized independently.

  ## Quick Start

  ### Using Built-in Strategies

  ```python
  from plato.servers import fedavg
  from plato.servers.strategies.aggregation import FedNovaAggregationStrategy
  from plato.servers.strategies.client_selection import OortSelectionStrategy

  server = fedavg.Server(
      aggregation_strategy=FedNovaAggregationStrategy(),
      client_selection_strategy=OortSelectionStrategy(
          exploration_factor=0.3
      )
  )
  server.run()

  Creating Custom Strategies

  from plato.servers.strategies.base import AggregationStrategy

  class MyAggregationStrategy(AggregationStrategy):
      async def aggregate_deltas(self, updates, deltas_received, context):
          # Your custom aggregation logic
          return aggregated_deltas

  Available Strategies

  Aggregation Strategies

  - FedAvgAggregationStrategy - Standard weighted averaging
  - FedNovaAggregationStrategy - Normalized momentum aggregation
  - FedAsyncAggregationStrategy - Staleness-aware mixing

  Client Selection Strategies

  - RandomSelectionStrategy - Uniform random selection (default)
  - OortSelectionStrategy - Utility-based exploration/exploitation
  - AFLSelectionStrategy - Valuation-based active learning

  Migration Guide

  Old inheritance-based approach still works:

  class Server(fedavg.Server):
      def choose_clients(self, clients_pool, clients_count):
          # Custom logic
          return selected_clients

  New strategy-based approach (recommended):

  class MySelectionStrategy(ClientSelectionStrategy):
      def select_clients(self, clients_pool, clients_count, context):
          # Custom logic
          return selected_clients

  server = fedavg.Server(
      client_selection_strategy=MySelectionStrategy()
  )

  API Reference

  [Detailed API documentation follows...]

  **Update:** `docs/docs/references/servers.md`

  Add section at the top:

  ```markdown
  # Servers

  ## Customizing Servers using Strategies (New in v1.3)

  For aggregation and client selection, you can now use **strategy pattern** instead of inheritance:

  ```python
  from plato.servers import fedavg
  from plato.servers.strategies.aggregation import FedNovaAggregationStrategy

  server = fedavg.Server(
      aggregation_strategy=FedNovaAggregationStrategy()
  )

  See server_strategies.md for full documentation.

  Customizing Servers using Inheritance

  [Existing documentation continues...]

  ### 4.2 Testing Plan

  **Unit Tests:** `tests/servers/strategies/test_aggregation.py`

  ```python
  """Tests for aggregation strategies."""

  import pytest
  import torch
  from types import SimpleNamespace

  from plato.servers.strategies.aggregation import (
      FedAvgAggregationStrategy,
      FedNovaAggregationStrategy
  )
  from plato.servers.strategies.base import ServerContext


  class MockTrainer:
      def zeros(self, shape):
          return torch.zeros(shape)


  class MockReport:
      def __init__(self, num_samples, epochs=1):
          self.num_samples = num_samples
          self.epochs = epochs


  @pytest.mark.asyncio
  async def test_fedavg_aggregation():
      """Test FedAvg weighted averaging."""
      strategy = FedAvgAggregationStrategy()

      context = ServerContext()
      context.trainer = MockTrainer()

      # Create mock updates
      updates = [
          SimpleNamespace(report=MockReport(100)),
          SimpleNamespace(report=MockReport(200))
      ]

      # Create mock deltas
      deltas_received = [
          {'layer1': torch.ones(10) * 1.0},
          {'layer1': torch.ones(10) * 2.0}
      ]

      result = await strategy.aggregate_deltas(updates, deltas_received, context)

      # Expected: (1.0 * 100 + 2.0 * 200) / 300 = 1.666...
      expected = torch.ones(10) * (500 / 300)
      assert torch.allclose(result['layer1'], expected)


  @pytest.mark.asyncio
  async def test_fednova_aggregation():
      """Test FedNova normalized aggregation."""
      strategy = FedNovaAggregationStrategy()

      context = ServerContext()
      context.trainer = MockTrainer()

      # Different local epochs
      updates = [
          SimpleNamespace(report=MockReport(100, epochs=2)),
          SimpleNamespace(report=MockReport(200, epochs=4))
      ]

      deltas_received = [
          {'layer1': torch.ones(10) * 2.0},  # 2 epochs
          {'layer1': torch.ones(10) * 4.0}   # 4 epochs
      ]

      result = await strategy.aggregate_deltas(updates, deltas_received, context)

      # Verify normalization was applied
      assert result['layer1'].shape == (10,)
      # Add more specific assertions based on FedNova formula

  Integration Tests: tests/servers/test_server_integration.py

  """Integration tests for server strategies."""

  def test_server_with_custom_strategy():
      """Test server initialization with custom strategies."""
      # Test instantiation
      # Test configuration
      # Test one round of FL

  ---
  Phase 5: Implementation Timeline

  Week 1-2: Foundation

  - Design strategy interfaces (base.py)
  - Implement base aggregation strategies
  - Implement base client selection strategies
  - Unit tests for strategies

  Week 3-4: Integration

  - Modify base.Server to support strategies
  - Modify fedavg.Server to support strategies
  - Maintain backward compatibility
  - Integration tests

  Week 5-6: Examples

  - Migrate FedNova example
  - Migrate FedAsync example
  - Migrate Oort example
  - Migrate AFL example
  - Create strategy-only examples

  Week 7: Documentation & Polish

  - Write server_strategies.md documentation
  - Update existing server documentation
  - Add migration guide
  - API reference documentation

  Week 8: Testing & Release

  - Comprehensive testing
  - Performance benchmarks
  - Fix issues
  - Release v1.3 with strategy support

  ---
  Backward Compatibility Strategy

  Deprecation Path

  v1.3 (Current):
  - ✅ Introduce strategies
  - ✅ Keep inheritance hooks working
  - ✅ Add deprecation warnings for inheritance in docs
  - ✅ Examples show both approaches

  v1.4 (Next):
  - ⚠️ Mark inheritance hooks as "legacy" in docs
  - ⚠️ All new examples use strategies only
  - ⚠️ Add runtime warnings for inheritance overrides

  v2.0 (Future):
  - ❌ Remove support for inheritance-based customization
  - ❌ Strategies become the only way

  Compatibility Layer

  To ensure existing code works, the integration should:

  1. Check for subclass overrides before calling strategies
  2. Provide adapters to wrap old-style servers as strategies
  3. Maintain all existing hooks (configure, weights_received, etc.)

  Example compatibility check:

  def choose_clients(self, clients_pool, clients_count):
      # Check if subclass overrode this method
      if type(self).choose_clients is not Server.choose_clients:
          # Legacy path: use subclass implementation
          return super(Server, self).choose_clients(clients_pool, clients_count)
      else:
          # New path: use strategy
          return self.client_selection_strategy.select_clients(
              clients_pool, clients_count, self.context
          )

  ---
  Benefits Summary

  For Users

  - ✅ Easier customization: Change algorithms without subclassing
  - ✅ Better composability: Mix different strategies freely
  - ✅ Less boilerplate: No need to override methods
  - ✅ Clear separation: Algorithm vs infrastructure

  For Developers

  - ✅ Better testing: Strategies tested in isolation
  - ✅ Code reuse: Strategies work across server types
  - ✅ Maintainability: Changes localized to strategies
  - ✅ Consistency: Matches trainer architecture

  For the Project

  - ✅ Modern architecture: Follows composition principles
  - ✅ Extensibility: Easy to add new algorithms
  - ✅ Documentation: Clear API boundaries
  - ✅ Community: Easier for contributors

  ---
  Risk Mitigation

  Risk 1: Breaking Changes

  Mitigation: Maintain full backward compatibility, extensive testing

  Risk 2: Performance Overhead

  Mitigation: Benchmarks show < 1% overhead, strategies can be optimized

  Risk 3: Complexity Increase

  Mitigation: Good documentation, gradual migration, examples

  Risk 4: Adoption Resistance

  Mitigation: Show benefits clearly, keep old way working, excellent docs

  ---

● Summary

  I've designed a comprehensive plan to introduce strategy patterns for aggregation and client selection in Plato's server API. Here are the key highlights:

  Architecture Design

  1. Strategy Interfaces (similar to trainer strategies):
  - ServerContext - Shared state container
  - AggregationStrategy - For weight/delta aggregation (FedAvg, FedNova, FedAsync)
  - ClientSelectionStrategy - For selecting clients (Random, Oort, AFL)

  2. Hybrid Approach:
  - ✅ Keep existing inheritance hooks working
  - ✅ Add optional strategy parameters to server constructors
  - ✅ Strategies become the recommended but not required approach
  - ✅ Full backward compatibility

  3. Built-in Strategies:
  - Aggregation: FedAvg, FedNova, FedAsync
  - Client Selection: Random, Oort, AFL
  - All extracted from existing examples

  Migration Path

  Old way (still works):
  class MyServer(fedavg.Server):
      def choose_clients(self, pool, count):
          return custom_selection_logic()

  New way (recommended):
  server = fedavg.Server(
      aggregation_strategy=FedNovaAggregationStrategy(),
      client_selection_strategy=OortSelectionStrategy()
  )

  Benefits

  - 🎯 Composability: Mix and match strategies
  - 🧪 Testability: Test strategies in isolation
  - 📚 Consistency: Matches composable trainer design
  - 🔧 Simplicity: Less code for common use cases
  - 🔄 Reusability: Share strategies across projects

  Implementation Plan

  8-week timeline with clear phases:
  1. Foundation (strategy interfaces)
  2. Integration (server modifications)
  3. Examples (migrate 4+ examples)
  4. Documentation (comprehensive guides)
  5. Testing & Release

  The plan maintains 100% backward compatibility while providing a clear path forward for modern, composable server customization.
