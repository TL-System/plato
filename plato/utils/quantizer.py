"""
A quantizer that quantizes a maching learning model
from 32-bit floating point format to 8-bit integer format,
and its corresponding de-quantizer.
"""

from collections import namedtuple
from collections import OrderedDict


def quantize_model_weights(weights):
    """Quantize weights before sending."""
    quantized_weights = OrderedDict()
    for name, weight in weights.items():
        quantized_tensor = quantize_tensor(weight)
        quantized_weights[name] = quantized_tensor
    return quantized_weights


def dequantize_model_weights(quantized_weights):
    """De-quantize quantized weights."""
    dequantized_weights = OrderedDict()
    for name, quantized_weight in quantized_weights.items():
        dequantized_weight = quantized_weight.scale * (
            quantized_weight.tensor.float() - quantized_weight.zero_point)
        dequantized_weights[name] = dequantized_weight

    return dequantized_weights


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def quantize_tensor(tensor, num_bits=8):
    """Quantize a 32-bit floating point tensor."""
    qmin = -2.**(num_bits - 1)
    qmax = 2.**(num_bits - 1) - 1.
    min_val, max_val = tensor.min(), tensor.max()

    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0.0:
        scale = 0.001

    initial_zero_point = qmin - min_val / scale

    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_tensor = zero_point + tensor / scale
    q_tensor.clamp_(qmin, qmax).round_()
    q_tensor = q_tensor.round().char()
    return QTensor(tensor=q_tensor, scale=scale, zero_point=zero_point)
