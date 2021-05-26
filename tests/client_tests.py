"""
Testing a federated learning client.
"""
import asyncio
from plato.clients import simple


async def test_training(client):
    """ Testing the training loop on the client. """
    print("Testing training on the client.")

    report, weights = await client.train()

    print("Client model trained.")
    print(f"Report to be sent to the server: {report}")
    print(f"Model weights: {weights}")


def main():
    """ Starting a simple client. """
    client = simple.Client()
    client.client_id = 1
    client.configure()
    client.load_data()
    asyncio.run(test_training(client))


if __name__ == "__main__":
    main()
