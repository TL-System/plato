"""
Implements a Processor for unbatching MistNet PyTorch features into the dataset form.
"""
import logging
from typing import Any

import numpy as np
from plato.processors import base
import zstd
import pickle


class Processor(base.Processor):
    """
    Implements a Processor for compressing numpy array
    """
    def __init__(self, cr = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cr = cr

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for compressring numpy array
        """
        if type(data) == list:
            ret = []
            datashape_feature = data[0][0].shape
            datatype_feature = data[0][0].dtype
            ret.append((datashape_feature, datatype_feature))
            for logits, targets in data:
                datashape_target = targets.shape
                datatype_target = targets.dtype
                datacom_feature = zstd.compress(logits, self.cr)
                datacom_target = zstd.compress(targets, self.cr)
                ret.append((datacom_feature, datacom_target, datashape_target, datatype_target))
        else:
            ret = (data.shape, data.dtype, zstd.compress(data, self.cr))
        
        return ret
