"""The dataset used for experiments in ControlNet"""

from plato.config import Config
from plato.datasources import base

from dataset.dataset_celeba import CelebADataset
from dataset.dataset_coco import CoCoDataset
from dataset.dataset_fill50k import Fill50KDataset
from dataset.dataset_omniglot import Omniglot


class DataSource(base.DataSource):
    """The datasource class specifiedly used for ControlNet privacy study."""
