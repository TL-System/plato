"""
Testing a federated learning client.
"""
import os
import sys
import asyncio

# To import modules from the parent directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config import Config
from clients import SimpleClient


async def test_training(client):
    print("Testing training on the client.")

    report = await client.train()
    print(report)


def main():
    """Starting a client to connect to the server via WebSockets."""
    __ = Config()

    loop = asyncio.get_event_loop()
    coroutines = []
    client = SimpleClient()
    client.client_id = "1"
    client.configure()
    client.load_data()

    try:
        coroutines.append(test_training(client))

        loop.run_until_complete(asyncio.gather(*coroutines))

    except Exception as exception:
        print(exception)
        sys.exit()

    os.remove("./running_trainers")


if __name__ == "__main__":
    main()
