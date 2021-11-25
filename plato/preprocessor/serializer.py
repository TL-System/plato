"""Serializer for serializing preprocessed NumPy array into ByteArray for
transfer."""

from plato.preprocessor import reversable


class Preprocessor(reversable.Preprocessor):
    """Serializer class for serializing preprocessed NumPy array into ByteArray
    for transfer."""
    def __init__(self) -> None:
        """Constructor for Serializer"""
        super().__init__()

    def process(self, data):
        """Serializing NumPy Array into ByteArray"""
        raise NotImplementedError()

    def stream_process(self, iterator):
        """Serializing NumPy Array into ByteArray"""
        raise NotImplementedError()

    def unprocess(self, data):
        """Deserializing NumPy Array into ByteArray"""
        raise NotImplementedError()

    def stream_unprocess(self, iterator):
        """Deserializing NumPy Array into ByteArray"""
        raise NotImplementedError()
