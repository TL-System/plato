"""
Testing the environment of reinforcement learning.
"""

import asyncio
import logging
import websockets
from stable_baselines3.common.env_checker import check_env

from config import Config
from rl_agent import RLAgent
from rl_envs import FLEnv


def main():
    """Check the custom environment of federated learning."""
    __ = Config()

    rl_port = Config().server.port + Config().clients.total_clients + 1
    if Config().cross_silo:
        rl_port += Config().cross_silo.total_silos

    rl_agent = RLAgent(rl_port)
    rl_env = FLEnv(rl_agent)
    rl_agent.configure(rl_env)

    logging.info("Starting a RL agent on port %s...", rl_port)

    start_rl_agent = websockets.serve(rl_agent.serve,
                                      Config().server.address,
                                      rl_port,
                                      ping_interval=None,
                                      max_size=2**30)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_rl_agent)

    check_with_sb3_env_checker(rl_env)

    try_a_random_agent(rl_env)

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
