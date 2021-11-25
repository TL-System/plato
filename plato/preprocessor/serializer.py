"""Serializer for serializing preprocessed NumPy array into ByteArray for
transfer."""

import pickle
from plato.preprocessor import reversable


class Preprocessor(reversable.Preprocessor):
    """Serializer class for serializing preprocessed NumPy array into ByteArray
    for transfer."""
    def __init__(self):
        """Constructor for Serializer"""
        super().__init__()

    def process(self, data):
        """Serializing NumPy Array into ByteArray"""
        return pickle.dumps(data)

    def unprocess(self, data):
        """Deserializing ByteArray into NumPy Array"""
        return pickle.loads(data)
