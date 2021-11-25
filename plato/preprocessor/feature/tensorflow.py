"""Formatter for formatting tensorflow tensors into and back from NumPy array"""

from plato.preprocessor import reversable


class Preprocessor(reversable.Preprocessor):
    """Formatter for formatting tensorflow tensors into and back from NumPy array"""
    def __init__(self):
        """Constructor for Tensorflow formatter"""
        super().__init__()

    def process(self, data):
        """Format tensorflow tensor into numpy array"""
        return data

    def unprocess(self, data):
        """Format numpy array into tensorflow tensor"""
        return data
