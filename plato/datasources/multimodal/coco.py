"""
The MSCOCO dataset
"""

import os

from plato.config import Config
from plato.datasources.multimodal import multimodal_base


class DataSource(multimodal_base.MultiModalDataSource):
    """The COCO dataset."""
    def __init__(self):
        super().__init__()

        self.data_name = Config().data.dataname
        self.data_source = Config().data.datasource

        self.modality_names = ["image", "text"]

        _path = Config().data.data_path
        self._data_path_process(data_path=_path, base_data_name=self.data_name)

        base_data_path = self.mm_data_info["base_data_dir_path"]
        raw_data_name = self.data_name + "Raw"
        raw_data_path = os.path.join(base_data_path, raw_data_name)
        if not self._exist_judgement(raw_data_path):
            os.makedirs(raw_data_path, exist_ok=True)

        download_train_url = Config().data.download_train_url
        download_val_url = Config().data.download_val_url
        download_annotation_url = Config().data.download_annotation_url

        for raw_url in [
                download_train_url, download_val_url, download_annotation_url
        ]:
            self._download_arrange_data(download_url_address=raw_url,
                                        put_data_dir=raw_data_path)
