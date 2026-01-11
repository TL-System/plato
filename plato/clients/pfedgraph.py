"""
pFedGraph client implementation.

Uses the standard simple client pipeline.
"""

from plato.clients import simple


class Client(simple.Client):
    """A pFedGraph client using default composable strategies."""
