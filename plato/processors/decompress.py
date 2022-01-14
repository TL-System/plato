"""
Implements a Processor for unbatching MistNet PyTorch features into the dataset form.
"""
import logging
from typing import Any

import numpy as np
from plato.processors import base
import zstd

class Processor(base.Processor):
    """
    Implements a Processor for decompressing numpy array
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for compressring numpy array
        """
        if type(data) == list:
            ret = []
            datashape_feature = data[0][0]
            datatype_feature = data[0][1]
            for datacom_feature, datacom_target, datashape_target, datatype_target in data[1:]:
                datacom_feature = zstd.decompress(datacom_feature)
                datacom_target = zstd.decompress(datacom_target)
                datacom_feature = np.frombuffer(datacom_feature, datatype_feature).reshape(datashape_feature)
                datacom_target = np.frombuffer(datacom_target, datatype_target).reshape(datashape_target)
                ret.append((datacom_feature, datacom_target))
        else:
            shape, dtype, modelcom = data
            modelcom = zstd.decompress(modelcom)
            ret = np.frombuffer(modelcom, dtype).reshape(shape)
            
        return ret
