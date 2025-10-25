"""
Legacy entry point for the AFL server using the strategy-based API.

This module simply re-exports the server defined in ``afl_server`` to avoid
breaking older entry points.
"""

from afl_server import Server

__all__ = ["Server"]
