"""
Flowers102 is a 102 category dataset, consisting of 102 flower categories. 
Each class consists of between 40 and 258 images

See details on the website, 
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.
"""
import os
import json
import random

from torchvision import datasets
from auxfl.models import clip

from plato.config import Config
from plato.datasources import base


class Flowers102OneClass(datasets.Flowers102):
    """Warpping the Flower102 for stable diffusion."""

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        super().__init__(
            root,
            split,
            transform,
            target_transform,
            download,
        )

        # extract the specific class
        target_class = Config().algorithm.target_class

        with open(os.path.join("examples/fedti", "cat_to_name.json")) as json_file:
            self.id_to_class = json.load(json_file)
        self.class_to_idx = {
            label_class: int(label_id) - 1
            for label_id, label_class in self.id_to_class.items()
        }

        target_class_id = self.class_to_idx[target_class]

        # image_ids; self._labels
        # self._labels
        new_images = []
        new_labels = []
        new_images_filename = []

        for idx, label_id in enumerate(self._labels):
            if label_id == target_class_id:
                image_file = self._image_files[idx]
                filename = os.path.basename(image_file)

                new_images.append(image_file)
                new_labels.append(image_file)
                new_images_filename.append(filename)

        self._image_files = new_images
        self._labels = new_labels
        self.images_filename = new_images_filename

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)

        image_filename = self.images_filename[idx]

        return image, target, image_filename

    def get_base_folder(self):
        return self._base_folder


class Flowers102FewShot(datasets.Flowers102):
    """Warpping the OxfordIIITPet for creating few-shot dataset."""

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        super().__init__(
            root,
            split,
            transform,
            target_transform,
            download,
        )

        # extract the specific class
        n_shots = Config().algorithm.n_shots

        with open(os.path.join(self._base_folder, "cat_to_name.json")) as json_file:
            self.id_to_class = json.load(json_file)

        self.class_to_idx = {
            label_class: int(label_id) - 1
            for label_id, label_class in self.id_to_class.items()
        }

        # image_ids; self._labels
        # self._labels
        # collecting samples for each class
        class_to_shots_idx = {}

        new_images = []
        new_labels = []
        new_images_filename = []

        for class_id in self.id_to_class:
            class_id = int(class_id) - 1
            samples_idx = [
                idx for idx, label_id in enumerate(self._labels) if label_id == class_id
            ]
            class_to_shots_idx[class_id] = random.sample(samples_idx, n_shots)

            class_images = [
                self._image_files[idx] for idx in class_to_shots_idx[class_id]
            ]
            new_images.extend(class_images)

            new_labels.extend(
                [self._labels[idx] for idx in class_to_shots_idx[class_id]]
            )
            new_images_filename.extend(
                [os.path.basename(image) for image in class_images]
            )

        self._image_files = new_images
        self._labels = new_labels
        self.images_filename = new_images_filename

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)

        image_filename = self.images_filename[idx]

        return image, target

    def get_base_folder(self):
        return self._base_folder

    @property
    def targets(self):
        return self._labels

    @property
    def classes(self):
        return list(self.id_to_class.keys())


class Flowers102Plato(datasets.Flowers102):
    @property
    def targets(self):
        return self._labels

    @property
    def classes(self):
        return list(set(self._labels))


class CLIPDataSource(base.DataSource):
    """The data source for few-shot classification with CLIP."""

    def __init__(self, **kwargs):
        super().__init__()

        (
            _,
            train_preprocesser,
            eval_preprocesser,
        ) = clip.CLIP.get_pretrained_model(
            model_name=Config().algorithm.personalization.model_name,
            pretrained_dataset=Config().algorithm.personalization.pretrained_dataset,
        )

        _path = Config().params["data_path"]

        self.trainset = Flowers102FewShot(
            root=_path, split="test", download=True, transform=train_preprocesser
        )
        self.testset = Flowers102Plato(
            root=_path, split="train", download=True, transform=eval_preprocesser
        )

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)


class DataSource(base.DataSource):
    """The Flowers102 dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        _path = Config().params["data_path"]

        train_transform = (
            kwargs["train_transform"] if "train_transform" in kwargs else None
        )

        self.trainset = Flowers102OneClass(
            root=_path, split="test", download=True, transform=train_transform
        )

        self.testset = Flowers102OneClass(
            root=_path, split="train", download=True, transform=train_transform
        )

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)
