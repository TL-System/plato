"""Serializer for serializing preprocessed NumPy array into ByteArray for
transfer."""

import pickle
from plato.DataProcessor import base


class Serializer(base.DataProcessor):
    """Serializer class for serializing NumPy array into ByteArray
    for transfer."""
    def __init__(self):
        """Constructor for Serializer"""
        super().__init__()

    def process(self, data):
        """Serializing NumPy Array into ByteArray"""
        return pickle.dumps(data)


class Deserializer(base.DataProcessor):
    """Deserializer class for serializing ByteArray into NumPy array
    for transfer."""
    def __init__(self):
        """Constructor for Serializer"""
        super().__init__()

    def process(self, data):
        """Deserializing ByteArray into NumPy Array"""
        return pickle.loads(data)
