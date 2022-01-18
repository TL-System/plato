"""
Implements a Processor for compressing a numpy array.
"""
from typing import Any

import zstd
from plato.processors import base


class Processor(base.Processor):
    """ Implements a Processor for compressing numpy array. """
    def __init__(self, cr=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.compression_ratio = cr

    def process(self, data: Any) -> Any:
        """ Implements a Processor for compressing numpy array. """
        if isinstance(data, list):
            ret = []
            datashape_feature = data[0][0].shape
            datatype_feature = data[0][0].dtype
            ret.append((datashape_feature, datatype_feature))
            for logits, targets in data:
                datashape_target = targets.shape
                datatype_target = targets.dtype
                datacom_feature = zstd.compress(logits, self.compression_ratio)
                datacom_target = zstd.compress(targets, self.compression_ratio)
                ret.append((datacom_feature, datacom_target, datashape_target,
                            datatype_target))
        else:
            ret = (data.shape, data.dtype,
                   zstd.compress(data, self.compression_ratio))

        return ret
