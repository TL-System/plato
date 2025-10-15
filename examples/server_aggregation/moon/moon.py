"""
Entry point for running the MOON server aggregation example.
"""

from __future__ import annotations

import moon_client
import moon_server
from moon_model import Model as MoonModel


def main():
    """Launch a Plato training session with the MOON algorithm."""
    model = MoonModel
    client = moon_client.Client(model=model)
    server = moon_server.Server(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
