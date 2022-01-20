"""
Implements a Processor for decompressing a numpy array.
"""
from typing import Any

import numpy as np
import zstd
from plato.processors import base


class Processor(base.Processor):
    """ Implements a Processor for decompressing a numpy array. """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:
        """ Implements a Processor for decompressing a numpy array. """
        if isinstance(data, list):
            ret = []
            datashape_feature = data[0][0]
            datatype_feature = data[0][1]
            for datacom_feature, datacom_target, datashape_target, datatype_target in data[1:]:
                datacom_feature = zstd.decompress(datacom_feature)
                datacom_feature = np.frombuffer(datacom_feature, datatype_feature).reshape(datashape_feature)
                if len(datashape_target) > 0 and datashape_target[0] == 0:
                    datacom_target = np.zeros(datashape_target)
                else:
                    datacom_target = zstd.decompress(datacom_target)
                    datacom_target = np.frombuffer(datacom_target, datatype_target).reshape(datashape_target)
                ret.append((datacom_feature, datacom_target))
        else:
            shape, dtype, modelcom = data
            modelcom = zstd.decompress(modelcom)
            ret = np.frombuffer(modelcom, dtype).reshape(shape)
            
        return ret
