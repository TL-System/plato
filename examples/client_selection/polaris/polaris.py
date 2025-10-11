"""
A federated learning training session using the Polaris client selection method.
"""

import os

import polaris_server


def main():
    """A Plato federated learning training session using the Polaris algorithm."""
    server = polaris_server.Server()
    server.run()


if __name__ == "__main__":
    main()
