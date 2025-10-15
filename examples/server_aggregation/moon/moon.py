"""
Entry point for running the MOON server aggregation example.
"""

from __future__ import annotations

import moon_client
import moon_server


def main():
    """Launch a Plato training session with the MOON algorithm."""
    client = moon_client.Client()
    server = moon_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
