"""
A FedAvg training session with customized models.
"""

import dlg_model
from plato.servers import fedavg
from plato.clients import simple


def main():
    """A FedAvg training session with customized models."""
    model = dlg_model.get()
    client = simple.Client(model=model)
    server = fedavg.Server(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
