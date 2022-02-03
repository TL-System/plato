"""
A federated learning server using Port.

Reference:

"How Asynchronous can Federated Learning Be?"

"""

import port_server


def main():
    """ A Plato federated learning training session using FedAsync. """
    server = port_server.Server()
    server.run()


if __name__ == "__main__":
    main()
