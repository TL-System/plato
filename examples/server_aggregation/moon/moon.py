"""
Entry point for running the MOON server aggregation example.
"""

from __future__ import annotations

import moon_client
import moon_server
from moon_model_factory import resolve_moon_model


def main():
    """Launch a Plato training session with the MOON algorithm."""
    model = resolve_moon_model()
    client = moon_client.create_client(model=model)
    server = moon_server.Server(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
