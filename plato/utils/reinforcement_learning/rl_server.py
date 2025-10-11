"""
A federated learning server with RL Agent.
"""

import asyncio
import logging
from abc import abstractmethod

from plato.servers import fedavg


class RLServer(fedavg.Server):
    """A federated learning server with an RL Agent."""

    def __init__(
        self,
        agent,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.agent = agent

    def reset(self):
        """Resetting the model, trainer, and algorithm on the server."""
        logging.info(
            "Reconfiguring the server for episode %d", self.agent.current_episode
        )

        self.model = None
        self.trainer = None
        self.algorithm = None
        self.init_trainer()

        self.current_round = 0

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using smart weighting."""
        self.update_state()

        # Extract the total number of samples
        num_samples = [update.report.num_samples for update in updates]
        self.total_samples = sum(num_samples)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        # e.g., wait for the new action from RL agent
        # if the action affects the global aggregation
        self.agent.num_samples = num_samples
        await self.agent.prep_agent_update()
        await self.update_action()

        # Use adaptive weighted average
        for i, update in enumerate(deltas_received):
            for name, delta in update.items():
                if delta.type() == "torch.LongTensor":
                    avg_update[name] += delta * self.smart_weighting[i][0]
                else:
                    avg_update[name] += delta * self.smart_weighting[i]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    async def update_action(self):
        """Updating the RL agent's actions."""
        if self.agent.current_step == 0:
            logging.info("[RL Agent] Preparing initial action...")
            self.agent.prep_action()
        else:
            await self.agent.action_updated.wait()
            self.agent.action_updated.clear()

        self.apply_action()

    def update_state(self):
        """Wrap up the state update to RL Agent."""
        # Pass new state to RL Agent
        self.agent.new_state = self.prep_state()
        self.agent.process_env_update()

    async def wrap_up(self) -> None:
        """Wrapping up when each round of training is done."""
        self.save_to_checkpoint()

        if self.agent.reset_env:
            self.agent.reset_env = False
            self.reset()
        if self.agent.finished:
            await self._close()

    @abstractmethod
    def prep_state(self):
        """Wrap up the state update to RL Agent."""
        return

    @abstractmethod
    def apply_action(self):
        """Apply action update from RL Agent to FL Env."""
