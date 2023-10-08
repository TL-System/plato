"""
The COCO dataset or other datasets for the YOLOv8 model.

For more information about COCO 128, which contains the first 128 images of the
COCO 2017 dataset, refer to https://www.kaggle.com/ultralytics/coco128.

For more information about the COCO 2017 dataset, refer to http://cocodataset.org.
"""

from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """The YOLO dataset."""

    # pylint: disable=unused-argument
    def __init__(self, **kwargs):
        super().__init__()

        self.grid_size = Config().parameters.grid_size
        self.data = check_det_dataset(Config().data.data_params)
        self.train_set = None
        self.test_set = None

    def get_train_set(self):
        single_class = Config().parameters.model.num_classes == 1

        if self.train_set is None:
            self.train_set = YOLODataset(
                img_path=self.data["train"],
                imgsz=Config().data.image_size,
                batch_size=Config().trainer.batch_size,
                augment=False,
                hyp=DEFAULT_CFG,
                rect=False,
                cache=False,
                single_cls=single_class,
                stride=int(self.grid_size),
                pad=0.0,
                prefix="",
                use_segments=False,
                use_keypoints=False,
                classes=Config().data.classes,
                data=self.data,
            )

        return self.train_set

    def get_test_set(self):
        single_class = Config().parameters.model.num_classes == 1

        if self.test_set is None:
            self.test_set = YOLODataset(
                img_path=self.data["val"],
                imgsz=Config().data.image_size,
                batch_size=Config().trainer.batch_size,
                augment=True,
                hyp=DEFAULT_CFG,
                rect=False,
                cache=False,
                single_cls=single_class,
                stride=int(self.grid_size),
                pad=0.0,
                prefix="",
                use_segments=False,
                use_keypoints=False,
                classes=Config().data.classes,
                data=self.data,
            )

        return self.test_set

    def num_train_examples(self):
        return Config().data.num_train_examples

    def num_test_examples(self):
        return Config().data.num_test_examples

    def classes(self):
        """Obtains a list of class names in the dataset."""
        return Config().data.classes
