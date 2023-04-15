"""Auxiliary training routines.

Usecases:
* Simulate models trained to arbitrary degrees
* Maliciously train model parameters to leak more information
"""

from .VAE import train_encoder_decoder

__all__ = ["train_encoder_decoder"]
