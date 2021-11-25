"""Formatter for formatting pytorch tensors into and back from NumPy array"""

from plato.preprocessor import reversable


class Preprocessor(reversable.Preprocessor):
    """Formatter for formatting pytorch tensors into and back from NumPy array"""
    def __init__(self):
        """Constructor for pytorch formatter"""
        super().__init__()

    def process(self, data):
        """Format pytorch tensor into numpy array"""
        return data

    def unprocess(self, data):
        """Format numpy array into pytorch tensor"""
        return data
