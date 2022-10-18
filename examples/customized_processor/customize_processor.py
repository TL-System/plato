"""
Customize the inbound and outbound processors through client callbacks.
"""

from customize_callback import CustomizeProcessorCallback

from plato.servers import fedavg
from plato.clients import simple


def main():
    """A Plato federated learning training session using CustomizeProcessorCallback."""
    client = simple.Client(callbacks=[CustomizeProcessorCallback])
    server = fedavg.Server()
    server.run(client)


if __name__ == "__main__":
    main()
