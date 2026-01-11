"""
pFedGraph algorithm wrapper.

pFedGraph uses standard FedAvg weight handling with a personalized aggregation
strategy on the server side.
"""

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """pFedGraph reuses the FedAvg algorithm primitives."""
