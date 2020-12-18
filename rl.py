"""
Starting point of a reinforcement learning agent.
This agent will tune a parameter of federated learning.
"""
# pylint: disable=E1101

import asyncio
from stable_baselines3.common.env_checker import check_env

from config import Config
import server as start_fl
import servers
from rl_agent import RLAgent
from rl_envs import FLEnv


def main():
    """Check the custom environment of federated learning."""
    __ = Config()

    rl_agent = RLAgent()
    rl_env = FLEnv(rl_agent)

    fl_server = {"fedavg": servers.fedavg.FedAvgServer}[Config().server.type]()
    rl_agent.configure(rl_env, fl_server)
    fl_server.register_rl_agent(rl_agent)
    fl_server.configure()

    start_fl.start_server_and_clients(fl_server)

    # Run the RL agent as a coroutine of FL central server
    coroutines = []
    #coroutines.append(rl_agent.wait_until_central_server_start())
    coroutines.append(rl_agent.serve())
    asyncio.gather(*coroutines)

    # Test the environment of reinforcement learning.
    check_with_sb3_env_checker(rl_env)
    #try_a_random_agent(rl_env)

    loop = asyncio.get_event_loop()
    loop.run_forever()


def check_with_sb3_env_checker(env):
    """
    Use helper provided by stable_baselines3
    to check that the environment runs without error.
    """
    # It will check the environment and output additional warnings if needed
    check_env(env)


def try_a_random_agent(env):
    """Quickly try a random agent on the environment."""
    obs = env.reset()
    n_steps = Config().rl.episodes
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
