"""
A federated learning training session using FEI.
"""

import logging

import fei_agent
import fei_client
import fei_server


def main():
    """A Plato federated learning training session using the FEI algorithm."""
    logging.info("Starting RL Environment's process.")
    client = fei_client.Client()
    agent = fei_agent.RLAgent()
    server = fei_server.RLServer(agent=agent)
    server.run(client)


if __name__ == "__main__":
    main()
