"""
A generic data source wrapping datasets provided by torchvision.

This module mirrors the flexibility of the Hugging Face datasource by allowing
runtime selection of any torchvision dataset together with configurable split
parameters and constructor arguments taken from the experiment configuration.
"""

from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    """Convert config-provided structures into regular dictionaries."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "__dict__"):
        return {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_") and not callable(val)
        }
    raise TypeError(
        f"Unsupported mapping type for torchvision datasource: {type(value)}"
    )


def _to_plain_list(value: Any) -> List[Any]:
    """Convert config-provided list-like data into a Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return list(value)
    raise TypeError("Expected a list-like object for dataset arguments.")


def _default_transform():
    """Factory providing a fresh default transform for image datasets."""
    return transforms.ToTensor()


def _celeba_target_transform(label):
    """Normalize CelebA targets while honouring configured attribute/identity flags."""
    if not isinstance(label, tuple):
        return label

    config = Config()
    data_cfg = getattr(config, "data", None)
    targets_cfg = getattr(data_cfg, "celeba_targets", None)

    attr_enabled = True
    identity_enabled = True

    if targets_cfg is not None:
        attr_enabled = bool(getattr(targets_cfg, "attr", False))
        identity_enabled = bool(getattr(targets_cfg, "identity", False))

        if not attr_enabled and not identity_enabled:
            # Fall back to legacy behaviour if both targets were disabled.
            attr_enabled = True
            identity_enabled = True

    pieces = list(label)
    attr_value = pieces[0] if pieces else None
    identity_value = pieces[-1] if pieces else None

    if attr_enabled and identity_enabled:
        attr_tensor = torch.as_tensor(attr_value).reshape(-1)
        identity_tensor = torch.as_tensor(identity_value).reshape(-1)
        return torch.cat((attr_tensor, identity_tensor))

    if identity_enabled:
        if isinstance(identity_value, torch.Tensor):
            if identity_value.numel() == 1:
                return identity_value.item()
            return identity_value.squeeze()
        return identity_value

    if attr_enabled:
        if isinstance(attr_value, torch.Tensor):
            return attr_value.reshape(-1)
        return attr_value

    return label


def _dataset_defaults(dataset_name: str, data_cfg) -> Dict[str, Any]:
    """Per-dataset defaults mirroring legacy torchvision-backed datasources."""

    name = dataset_name.lower()

    if name == "mnist":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        return {"train_transform": transform, "test_transform": transform}

    if name == "fashionmnist":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        return {"train_transform": transform, "test_transform": transform}

    if name == "emnist":
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        return {
            "train_transform": train_transform,
            "test_transform": test_transform,
            "dataset_kwargs": {"split": "balanced"},
        }

    if name in {"cifar10", "cifar100"}:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return {"train_transform": train_transform, "test_transform": test_transform}

    if name == "stl10":
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        return {
            "train_transform": train_transform,
            "test_transform": test_transform,
            "unlabeled_transform": train_transform,
        }

    if name == "celeba":
        target_types: List[str] = []
        if data_cfg is not None and hasattr(data_cfg, "celeba_targets"):
            targets_cfg = data_cfg.celeba_targets
            if getattr(targets_cfg, "attr", False):
                target_types.append("attr")
            if getattr(targets_cfg, "identity", False):
                target_types.append("identity")
        if not target_types:
            target_types = ["attr", "identity"]

        image_size = 64
        if data_cfg is not None and hasattr(data_cfg, "celeba_img_size"):
            image_size = getattr(data_cfg, "celeba_img_size")

        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        defaults: Dict[str, Any] = {
            "train_transform": transform,
            "test_transform": transform,
            "dataset_kwargs": {"target_type": target_types},
        }

        if target_types:
            defaults["train_target_transform"] = _celeba_target_transform
            defaults["test_target_transform"] = _celeba_target_transform

        return defaults

    return {}


class DataSource(base.DataSource):
    """A datasource capable of loading any dataset exposed by torchvision."""

    def __init__(self, **kwargs):
        super().__init__()

        config = Config()
        data_cfg = config.data

        dataset_name_override = kwargs.pop("dataset_name", None)
        dataset_name = getattr(data_cfg, "dataset_name", dataset_name_override)
        if dataset_name is None:
            raise ValueError(
                "`dataset_name` must be specified for the Torchvision datasource."
            )

        if not hasattr(data_cfg, "dataset_name"):
            setattr(data_cfg, "dataset_name", dataset_name)

        logging.info("Torchvision dataset: %s", dataset_name)

        dataset_cls = self._resolve_dataset_class(dataset_name)
        signature = inspect.signature(dataset_cls.__init__)

        dataset_defaults = _dataset_defaults(dataset_name, data_cfg)

        split_parameter = self._determine_split_parameter(signature, data_cfg)
        default_train, default_test = self._default_split_values(split_parameter)

        train_split = self._normalize_split_value(
            split_parameter, getattr(data_cfg, "train_split", None), default_train
        )
        test_split = self._normalize_split_value(
            split_parameter, getattr(data_cfg, "test_split", None), default_test
        )

        if hasattr(data_cfg, "unlabeled_split"):
            unlabeled_split = data_cfg.unlabeled_split
        elif dataset_name.lower() == "stl10" and split_parameter == "split":
            unlabeled_split = "unlabeled"
        elif "unlabeled_split" in dataset_defaults:
            unlabeled_split = dataset_defaults["unlabeled_split"]
        else:
            unlabeled_split = None

        user_common_kwargs = _to_plain_dict(getattr(data_cfg, "dataset_kwargs", None))
        user_train_kwargs = _to_plain_dict(getattr(data_cfg, "train_kwargs", None))
        user_test_kwargs = _to_plain_dict(getattr(data_cfg, "test_kwargs", None))
        user_unlabeled_kwargs = _to_plain_dict(
            getattr(data_cfg, "unlabeled_kwargs", None)
        )

        common_args = list(dataset_defaults.get("dataset_args", [])) + _to_plain_list(
            getattr(data_cfg, "dataset_args", None)
        )
        train_args = list(dataset_defaults.get("train_args", [])) + _to_plain_list(
            getattr(data_cfg, "train_args", None)
        )
        test_args = list(dataset_defaults.get("test_args", [])) + _to_plain_list(
            getattr(data_cfg, "test_args", None)
        )
        unlabeled_args = list(
            dataset_defaults.get("unlabeled_args", [])
        ) + _to_plain_list(getattr(data_cfg, "unlabeled_args", None))

        common_kwargs = {
            **dataset_defaults.get("dataset_kwargs", {}),
            **user_common_kwargs,
        }
        train_kwargs = {
            **dataset_defaults.get("train_kwargs", {}),
            **user_train_kwargs,
        }
        test_kwargs = {
            **dataset_defaults.get("test_kwargs", {}),
            **user_test_kwargs,
        }
        unlabeled_kwargs = {
            **dataset_defaults.get("unlabeled_kwargs", {}),
            **user_unlabeled_kwargs,
        }

        common_kwargs.setdefault("root", config.params["data_path"])

        download_flag = getattr(data_cfg, "download", kwargs.get("download", True))
        download_supported = "download" in signature.parameters

        default_train_transform = dataset_defaults.get("train_transform")
        if default_train_transform is None:
            default_train_transform = _default_transform()
        default_test_transform = dataset_defaults.get("test_transform")
        if default_test_transform is None:
            default_test_transform = _default_transform()

        train_transform = kwargs.get("train_transform", default_train_transform)
        test_transform = kwargs.get("test_transform", default_test_transform)

        default_train_target_transform = dataset_defaults.get("train_target_transform")
        default_test_target_transform = dataset_defaults.get("test_target_transform")
        default_unlabeled_target_transform = dataset_defaults.get(
            "unlabeled_target_transform"
        )

        default_unlabeled_transform = dataset_defaults.get(
            "unlabeled_transform", train_transform
        )
        unlabeled_transform = kwargs.get(
            "unlabeled_transform", default_unlabeled_transform
        )

        train_target_transform = kwargs.get(
            "train_target_transform", default_train_target_transform
        )
        test_target_transform = kwargs.get(
            "test_target_transform", default_test_target_transform
        )
        unlabeled_target_transform = kwargs.get(
            "unlabeled_target_transform", default_unlabeled_target_transform
        )

        load_train = getattr(data_cfg, "load_train", True)
        load_test = getattr(data_cfg, "load_test", True)

        if load_train:
            self.trainset = self._instantiate_dataset(
                dataset_cls,
                common_args,
                train_args,
                common_kwargs,
                train_kwargs,
                split_parameter,
                train_split,
                download_flag if download_supported else None,
                train_transform,
                train_target_transform,
            )
        else:
            self.trainset = None

        if load_test:
            test_download = None
            if download_supported and "download" not in test_kwargs:
                test_download = download_flag if not load_train else False

            self.testset = self._instantiate_dataset(
                dataset_cls,
                common_args,
                test_args,
                common_kwargs,
                test_kwargs,
                split_parameter,
                test_split,
                test_download,
                test_transform,
                test_target_transform,
            )
        else:
            self.testset = None

        if unlabeled_split is not None:
            unlabeled_download = None
            if download_supported and "download" not in unlabeled_kwargs:
                unlabeled_download = False

            self.unlabeledset = self._instantiate_dataset(
                dataset_cls,
                common_args,
                unlabeled_args,
                common_kwargs,
                unlabeled_kwargs,
                split_parameter,
                unlabeled_split,
                unlabeled_download,
                unlabeled_transform,
                unlabeled_target_transform,
            )
        else:
            self.unlabeledset = None

        if self.trainset is None and self.testset is None:
            raise ValueError(
                "Torchvision datasource requires at least one split to be loaded."
            )

    @staticmethod
    def _resolve_dataset_class(dataset_name: str):
        """Locate the requested dataset class from torchvision.datasets."""
        if hasattr(datasets, dataset_name):
            candidate = getattr(datasets, dataset_name)
            if callable(candidate):
                return candidate

        normalized = dataset_name.lower()
        for name in dir(datasets):
            candidate = getattr(datasets, name)
            if name.lower() == normalized and callable(candidate):
                return candidate

        raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets.")

    @staticmethod
    def _determine_split_parameter(signature, data_cfg) -> Optional[str]:
        """Determine which constructor parameter controls data splits."""
        if hasattr(data_cfg, "split_parameter"):
            requested = data_cfg.split_parameter
            if requested not in signature.parameters:
                raise ValueError(
                    f"The dataset does not define a `{requested}` parameter for splits."
                )
            return requested

        if "train" in signature.parameters:
            return "train"
        if "split" in signature.parameters:
            return "split"
        return None

    @staticmethod
    def _default_split_values(split_parameter: Optional[str]):
        """Provide default split values based on the parameter type."""
        if split_parameter == "train":
            return True, False
        if split_parameter == "split":
            return "train", "test"
        return None, None

    @staticmethod
    def _normalize_split_value(
        split_parameter: Optional[str], value: Any, default: Any
    ):
        """Normalise split configuration values to the expected types."""
        if split_parameter is None:
            return None
        if value is None:
            return default
        if split_parameter == "train":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in {"train", "true", "1", "yes"}:
                    return True
                if lowered in {"test", "false", "0", "no"}:
                    return False
                raise ValueError(f"Unable to map '{value}' to a boolean split value.")
            if isinstance(value, (int, float)):
                return bool(value)
            raise TypeError(f"Unsupported type for train split: {type(value)}")
        return value

    def _instantiate_dataset(
        self,
        dataset_cls,
        shared_args: List[Any],
        extra_args: List[Any],
        shared_kwargs: Dict[str, Any],
        extra_kwargs: Dict[str, Any],
        split_parameter: Optional[str],
        split_value: Any,
        download: Optional[bool],
        transform: Any,
        target_transform: Any,
    ):
        """Instantiate a dataset split with merged positional and keyword args."""
        args = list(shared_args)
        args.extend(extra_args)

        kwargs = deepcopy(shared_kwargs)
        kwargs.update(extra_kwargs)

        if (
            split_parameter
            and split_parameter not in kwargs
            and split_value is not None
        ):
            kwargs[split_parameter] = split_value

        if transform is not None and "transform" not in kwargs:
            kwargs["transform"] = transform
        if target_transform is not None and "target_transform" not in kwargs:
            kwargs["target_transform"] = target_transform
        if download is not None and "download" not in kwargs:
            kwargs["download"] = download

        dataset = dataset_cls(*args, **kwargs)
        self._attach_metadata(dataset)
        return dataset

    @staticmethod
    def _attach_metadata(dataset):
        """Ensure standard attributes are available for downstream components."""
        class_name = dataset.__class__.__name__.lower()

        if class_name == "celeba" and hasattr(dataset, "identity"):
            identities = dataset.identity.reshape(-1)
            dataset.targets = identities.tolist()
            if not hasattr(dataset, "classes") or dataset.classes is None:
                try:
                    max_identity = int(identities.max().item())
                except Exception:  # pylint: disable=broad-except
                    max_identity = len(dataset.targets) - 1
                dataset.classes = [f"Celebrity #{i}" for i in range(max_identity + 1)]

        if not hasattr(dataset, "targets") and hasattr(dataset, "labels"):
            dataset.targets = dataset.labels
        if not hasattr(dataset, "classes") and hasattr(dataset, "class_to_idx"):
            dataset.classes = list(dataset.class_to_idx.keys())

    def classes(self):
        dataset = self.trainset or self.testset
        if dataset is None:
            return []
        if hasattr(dataset, "classes") and dataset.classes is not None:
            return list(dataset.classes)
        if hasattr(dataset, "class_to_idx"):
            return list(dataset.class_to_idx.keys())
        return []

    def targets(self):
        dataset = self.trainset or self.testset
        if dataset is None:
            return []
        if hasattr(dataset, "targets"):
            return dataset.targets
        if hasattr(dataset, "labels"):
            return dataset.labels
        return []

    def get_unlabeled_set(self):
        return getattr(self, "unlabeledset", None)

    @staticmethod
    def input_shape():
        raise ValueError("Input shape depends on the selected torchvision dataset.")
