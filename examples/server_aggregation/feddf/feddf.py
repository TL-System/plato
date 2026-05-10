"""
Entry point for running the FedDF server aggregation example.
"""

from __future__ import annotations

import feddf_client
import feddf_server


def main():
    """Launch a Plato training session with the FedDF algorithm."""
    client = feddf_client.create_client()
    server = feddf_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
