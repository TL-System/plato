"""LeRobot datasource with deterministic, episode-aware client partitioning."""

from __future__ import annotations

import hashlib
import inspect
import logging
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from plato.config import Config
from plato.datasources import base

_DEFAULT_TRAIN_SPLIT = 0.8
_DEFAULT_SPLIT_SEED = 1
_DEFAULT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_NORMALIZE_STD = (0.229, 0.224, 0.225)

_EPISODE_INDEX_KEYS = ("episode_index", "episode_id", "index")
_TASK_KEYS = (
    "task",
    "task_name",
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
    "task_id",
)


class _EmptyDataset:
    """A minimal dataset object for empty episode partitions."""

    targets: list[Any] = []
    classes: list[str] = []

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError(f"Empty dataset does not contain index {index}.")


class _MappedLeRobotDataset:
    """Wrap a LeRobot dataset and attach Plato-friendly canonical keys."""

    def __init__(self, dataset: Any):
        self._dataset = dataset
        self.targets = getattr(dataset, "targets", None)
        self.classes = getattr(dataset, "classes", None)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._dataset[index]

        if isinstance(sample, Mapping):
            raw_sample = dict(sample)
        else:
            return {
                "plato_inputs": sample,
                "plato_targets": None,
                "plato_metadata": {},
            }

        inputs: dict[str, Any] = {}
        targets: dict[str, Any] = {}
        metadata: dict[str, Any] = {}

        for key, value in raw_sample.items():
            if key.startswith("observation"):
                inputs[key] = value
            elif key == "action" or key.startswith("action."):
                targets[key] = value
            else:
                metadata[key] = value

        mapped = dict(raw_sample)
        mapped["plato_inputs"] = inputs
        mapped["plato_targets"] = raw_sample.get("action", targets or None)
        mapped["plato_metadata"] = metadata
        return mapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


def _import_lerobot() -> tuple[Any, Any]:
    try:
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError(
            "LeRobot datasource requires optional LeRobot / SmolVLA robotics dependencies. "
            "Install the robotics stack in the active environment before using "
            '"data.datasource = \"LeRobot\"". '
        ) from exc

    return LeRobotDataset, LeRobotDatasetMetadata


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _to_plain(val) for key, val in value.items()}
    if hasattr(value, "_asdict"):
        return {str(key): _to_plain(val) for key, val in value._asdict().items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item) for item in value]
    return value


def _to_plain_dict(value: Any) -> dict[str, Any]:
    plain = _to_plain(value)
    if plain is None:
        return {}
    if isinstance(plain, Mapping):
        return {str(key): val for key, val in plain.items()}
    raise TypeError(f"Expected mapping-like configuration, got {type(value)}.")


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _parse_size(value: Any) -> tuple[int, int] | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [int(item) for item in value]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return (parts[0], parts[1])
    raise TypeError("Expected an int or sequence for transform size.")


def _parse_float_sequence(value: Any, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = [float(item) for item in value]
        if parsed:
            return tuple(parsed)
    raise TypeError("Expected a numeric sequence for normalization values.")


def _resolve_interpolation(value: Any) -> Any:
    if value is None:
        return None

    try:
        from torchvision.transforms import InterpolationMode
    except ImportError:
        return None

    interpolation_map = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }
    return interpolation_map.get(str(value).strip().lower(), None)


def _build_image_transforms(transform_cfg: Mapping[str, Any]) -> Any | None:
    if not transform_cfg:
        return None

    enable = transform_cfg.get("enable", True)
    if isinstance(enable, bool) and not enable:
        return None

    try:
        import torch
        from torchvision import transforms as tv_transforms
    except ImportError as exc:
        raise ImportError(
            "LeRobot image transforms require torchvision and torch."
        ) from exc

    transform_steps: list[Any] = []

    image_size = _parse_size(transform_cfg.get("image_size"))
    interpolation = _resolve_interpolation(transform_cfg.get("interpolation"))

    if image_size is not None:
        resize_kwargs: dict[str, Any] = {}
        if interpolation is not None:
            resize_kwargs["interpolation"] = interpolation
        transform_steps.append(tv_transforms.Resize(image_size, **resize_kwargs))

    center_crop_cfg = transform_cfg.get("center_crop", None)
    crop_size = None
    if isinstance(center_crop_cfg, bool):
        if center_crop_cfg and image_size is not None:
            crop_size = image_size
    elif center_crop_cfg is not None:
        crop_size = _parse_size(center_crop_cfg)
    elif transform_cfg.get("crop_size") is not None:
        crop_size = _parse_size(transform_cfg.get("crop_size"))

    if crop_size is not None:
        transform_steps.append(tv_transforms.CenterCrop(crop_size))

    normalize = bool(transform_cfg.get("normalize", False))
    if normalize:
        convert_dtype = getattr(tv_transforms, "ConvertImageDtype", None)
        if callable(convert_dtype):
            transform_steps.append(convert_dtype(torch.float32))

        mean = _parse_float_sequence(
            transform_cfg.get("mean"),
            _DEFAULT_NORMALIZE_MEAN,
        )
        std = _parse_float_sequence(transform_cfg.get("std"), _DEFAULT_NORMALIZE_STD)
        transform_steps.append(tv_transforms.Normalize(mean=mean, std=std))

    if not transform_steps:
        return None

    compose = getattr(tv_transforms, "Compose", None)
    if not callable(compose):
        return None

    return compose(transform_steps)


def _normalize_delta_timestamps(value: Any) -> dict[str, list[float]] | None:
    if value is None:
        return None

    parsed = _to_plain(value)
    if not isinstance(parsed, Mapping):
        raise TypeError(
            '"parameters.dataset.delta_timestamps" must be a mapping of '
            "key -> list[float]."
        )

    normalized: dict[str, list[float]] = {}
    for key, offsets in parsed.items():
        if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
            raise TypeError(
                f"Delta timestamps for '{key}' must be a list-like sequence."
            )
        normalized[str(key)] = [float(offset) for offset in offsets]

    return normalized


def _columnar_to_rows(columns: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized: dict[str, list[Any]] = {}
    max_len = 0

    for key, value in columns.items():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = list(value)
        elif hasattr(value, "tolist"):
            values = list(value.tolist())
        else:
            values = [value]

        normalized[str(key)] = values
        max_len = max(max_len, len(values))

    rows: list[dict[str, Any]] = []
    for row_idx in range(max_len):
        row: dict[str, Any] = {}
        for key, values in normalized.items():
            if row_idx < len(values):
                row[key] = values[row_idx]
        rows.append(row)

    return rows


def _episode_rows(episodes: Any) -> list[dict[str, Any]]:
    if episodes is None:
        return []

    if isinstance(episodes, Mapping):
        return _columnar_to_rows(episodes)

    to_pylist = getattr(episodes, "to_pylist", None)
    if callable(to_pylist):
        pylist = to_pylist()
        if isinstance(pylist, list):
            return [
                dict(item) if isinstance(item, Mapping) else {"value": item}
                for item in pylist
            ]

    to_dict = getattr(episodes, "to_dict", None)
    if callable(to_dict):
        try:
            as_dict = to_dict(orient="list")
        except TypeError:
            as_dict = to_dict()
        if isinstance(as_dict, Mapping):
            return _columnar_to_rows(as_dict)

    if isinstance(episodes, Sequence) and not isinstance(episodes, (str, bytes)):
        rows: list[dict[str, Any]] = []
        for entry in episodes:
            if isinstance(entry, Mapping):
                rows.append(dict(entry))
            elif hasattr(entry, "_asdict"):
                rows.append(_to_plain_dict(entry))
            else:
                rows.append({"value": entry})
        return rows

    return []


def _resolve_episode_indices(metadata: Any) -> list[int]:
    total_episodes = _as_int(getattr(metadata, "total_episodes", None))

    if total_episodes is None:
        total_episodes = len(_episode_rows(getattr(metadata, "episodes", None)))

    if total_episodes is None or total_episodes < 0:
        return []

    return list(range(total_episodes))


def _resolve_task_name(row: Mapping[str, Any], tasks_lookup: Any) -> str | None:
    task_index = _as_int(row.get("task_index"))
    if task_index is not None and isinstance(tasks_lookup, Sequence):
        if not isinstance(tasks_lookup, (str, bytes)) and 0 <= task_index < len(
            tasks_lookup
        ):
            return str(tasks_lookup[task_index])

    for key in _TASK_KEYS:
        if key in row and row[key] is not None:
            return str(row[key])

    return None


def _resolve_episode_tasks(metadata: Any, episodes: Sequence[int]) -> dict[int, str | None]:
    episode_tasks = {episode: None for episode in episodes}
    episode_rows = _episode_rows(getattr(metadata, "episodes", None))
    tasks_lookup = _to_plain(getattr(metadata, "tasks", None))

    for position, row in enumerate(episode_rows):
        episode_index = None
        for key in _EPISODE_INDEX_KEYS:
            if key in row:
                episode_index = _as_int(row.get(key))
                if episode_index is not None:
                    break

        if episode_index is None:
            episode_index = position

        if episode_index not in episode_tasks:
            continue

        task_name = _resolve_task_name(row, tasks_lookup)
        if task_name is not None:
            episode_tasks[episode_index] = task_name

    return episode_tasks


def _split_count(total: int, train_ratio: float) -> int:
    if total <= 1:
        return total

    if train_ratio <= 0:
        return 0
    if train_ratio >= 1:
        return total

    split_count = int(round(total * train_ratio))
    split_count = max(1, min(total - 1, split_count))
    return split_count


def _split_episodes(
    episodes: Sequence[int],
    episode_tasks: Mapping[int, str | None],
    train_ratio: float,
    seed: int,
    task_aware: bool,
) -> tuple[list[int], list[int]]:
    if not episodes:
        return [], []

    all_episodes = [int(episode) for episode in episodes]
    has_task_info = task_aware and any(episode_tasks.get(ep) for ep in all_episodes)

    if has_task_info:
        grouped: dict[str, list[int]] = defaultdict(list)
        for episode in all_episodes:
            grouped[str(episode_tasks.get(episode) or "__unknown_task__")].append(
                episode
            )

        train_episodes: list[int] = []
        test_episodes: list[int] = []

        for task_name in sorted(grouped):
            task_episodes = sorted(grouped[task_name])
            rng = random.Random(_stable_seed(seed, f"split:{task_name}"))
            rng.shuffle(task_episodes)

            split_idx = _split_count(len(task_episodes), train_ratio)
            train_episodes.extend(task_episodes[:split_idx])
            test_episodes.extend(task_episodes[split_idx:])
    else:
        shuffled = sorted(all_episodes)
        random.Random(seed).shuffle(shuffled)
        split_idx = _split_count(len(shuffled), train_ratio)
        train_episodes = shuffled[:split_idx]
        test_episodes = shuffled[split_idx:]

    if not train_episodes and test_episodes:
        train_episodes.append(test_episodes.pop(0))

    return sorted(train_episodes), sorted(test_episodes)


def _normalize_episode_list(value: Any) -> list[int] | None:
    if value is None:
        return None

    parsed = _to_plain(value)

    if isinstance(parsed, int):
        return [int(parsed)]

    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        return [int(item) for item in parsed]

    raise TypeError("Episode lists must be an int or a list of ints.")


def _resolve_episode_split(
    all_episodes: Sequence[int],
    explicit_train: list[int] | None,
    explicit_test: list[int] | None,
    episode_tasks: Mapping[int, str | None],
    train_ratio: float,
    seed: int,
    task_aware: bool,
) -> tuple[list[int], list[int]]:
    episode_set = set(int(episode) for episode in all_episodes)

    if explicit_train is None and explicit_test is None:
        return _split_episodes(all_episodes, episode_tasks, train_ratio, seed, task_aware)

    train_episodes = [
        int(episode) for episode in (explicit_train or []) if int(episode) in episode_set
    ]
    test_episodes = [
        int(episode)
        for episode in (explicit_test or [])
        if int(episode) in episode_set and int(episode) not in train_episodes
    ]

    if not train_episodes:
        train_episodes = [
            episode for episode in all_episodes if episode not in set(test_episodes)
        ]

    if not test_episodes:
        test_episodes = [
            episode for episode in all_episodes if episode not in set(train_episodes)
        ]

    if not train_episodes and test_episodes:
        train_episodes.append(test_episodes.pop(0))

    return sorted(train_episodes), sorted(test_episodes)


def _partition_episodes(
    episodes: Sequence[int],
    episode_tasks: Mapping[int, str | None],
    total_clients: int,
    client_id: int,
    seed: int,
    task_aware: bool,
) -> list[int]:
    if not episodes:
        return []

    if total_clients <= 1 or client_id <= 0:
        return sorted(int(episode) for episode in episodes)

    client_slot = (int(client_id) - 1) % int(total_clients)
    all_episodes = [int(episode) for episode in episodes]

    has_task_info = task_aware and any(episode_tasks.get(ep) for ep in all_episodes)

    selected: list[int] = []

    if has_task_info:
        grouped: dict[str, list[int]] = defaultdict(list)
        for episode in all_episodes:
            grouped[str(episode_tasks.get(episode) or "__unknown_task__")].append(
                episode
            )

        for task_name in sorted(grouped):
            task_episodes = sorted(grouped[task_name])
            rng = random.Random(_stable_seed(seed, f"partition:{task_name}"))
            rng.shuffle(task_episodes)

            for idx, episode in enumerate(task_episodes):
                if idx % total_clients == client_slot:
                    selected.append(episode)
    else:
        shuffled = sorted(all_episodes)
        random.Random(seed).shuffle(shuffled)
        selected = [
            episode
            for idx, episode in enumerate(shuffled)
            if idx % total_clients == client_slot
        ]

    if not selected:
        fallback = sorted(all_episodes)
        random.Random(_stable_seed(seed, "fallback")).shuffle(fallback)
        selected = [fallback[client_slot % len(fallback)]]

    return sorted(selected)


def _resolve_default_seed() -> int:
    data_cfg = getattr(Config(), "data", None)
    configured_seed = _as_int(getattr(data_cfg, "random_seed", None))
    if configured_seed is not None:
        return configured_seed
    return _DEFAULT_SPLIT_SEED


def _resolve_total_clients(config: Any) -> int:
    clients_cfg = getattr(config, "clients", None)
    total_clients = _as_int(getattr(clients_cfg, "total_clients", 1))
    if total_clients is None or total_clients <= 0:
        return 1
    return total_clients


def _filter_constructor_kwargs(dataset_cls: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(dataset_cls.__init__)
    except (TypeError, ValueError):
        return dict(kwargs)

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return dict(kwargs)

    valid_parameters = {
        name for name in signature.parameters.keys() if name != "self"
    }
    filtered = {key: value for key, value in kwargs.items() if key in valid_parameters}

    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        logging.warning(
            "LeRobot datasource ignored unsupported dataset kwargs: %s",
            ", ".join(dropped),
        )

    return filtered


class DataSource(base.DataSource):
    """LeRobot datasource with deterministic train/test and per-client episode splits."""

    def __init__(self, client_id: int = 0, **kwargs):
        super().__init__()

        LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot()

        config = Config()
        params_cfg = getattr(config, "parameters", None)

        dataset_cfg = _to_plain_dict(getattr(params_cfg, "dataset", None))
        transform_cfg = _to_plain_dict(getattr(params_cfg, "transforms", None))

        dataset_cfg.update(_to_plain_dict(kwargs.pop("dataset_kwargs", None)))
        transform_cfg.update(_to_plain_dict(kwargs.pop("transform_kwargs", None)))

        for key in (
            "repo_id",
            "delta_timestamps",
            "split_seed",
            "train_split",
            "test_split",
            "train_episodes",
            "test_episodes",
            "task_aware_split",
            "task_aware_partition",
        ):
            if key in kwargs:
                dataset_cfg[key] = kwargs.pop(key)

        for key in (
            "image_size",
            "interpolation",
            "center_crop",
            "crop_size",
            "normalize",
            "mean",
            "std",
            "enable",
        ):
            if key in kwargs:
                transform_cfg[key] = kwargs.pop(key)

        dataset_cfg.update(_to_plain_dict(kwargs))

        repo_id = str(dataset_cfg.pop("repo_id", "")).strip()
        if not repo_id:
            raise ValueError(
                "LeRobot datasource requires "
                '"parameters.dataset.repo_id" to be set.'
            )

        train_split_raw = dataset_cfg.pop("train_split", _DEFAULT_TRAIN_SPLIT)
        test_split_raw = dataset_cfg.pop("test_split", None)

        train_split = float(train_split_raw)
        if test_split_raw is not None:
            train_split = 1.0 - float(test_split_raw)
        train_split = max(0.0, min(1.0, train_split))

        split_seed = _as_int(dataset_cfg.pop("split_seed", None))
        if split_seed is None:
            split_seed = _resolve_default_seed()

        task_aware_split = bool(dataset_cfg.pop("task_aware_split", True))
        task_aware_partition = bool(dataset_cfg.pop("task_aware_partition", True))

        delta_timestamps = _normalize_delta_timestamps(
            dataset_cfg.pop("delta_timestamps", None)
        )

        explicit_train_episodes = _normalize_episode_list(
            dataset_cfg.pop("train_episodes", None)
        )
        explicit_test_episodes = _normalize_episode_list(
            dataset_cfg.pop("test_episodes", None)
        )

        metadata = LeRobotDatasetMetadata(repo_id)
        all_episodes = _resolve_episode_indices(metadata)

        if not all_episodes:
            raise ValueError(f"No episodes found for LeRobot dataset '{repo_id}'.")

        episode_tasks = _resolve_episode_tasks(metadata, all_episodes)
        train_episodes, test_episodes = _resolve_episode_split(
            all_episodes,
            explicit_train_episodes,
            explicit_test_episodes,
            episode_tasks,
            train_split,
            split_seed,
            task_aware_split,
        )

        total_clients = _resolve_total_clients(config)
        resolved_client_id = int(client_id)

        client_train_episodes = _partition_episodes(
            train_episodes,
            episode_tasks,
            total_clients,
            resolved_client_id,
            split_seed,
            task_aware_partition,
        )
        client_test_episodes = _partition_episodes(
            test_episodes,
            episode_tasks,
            total_clients,
            resolved_client_id,
            split_seed + 1,
            task_aware_partition,
        )

        image_transforms = _build_image_transforms(transform_cfg)
        dataset_kwargs = _filter_constructor_kwargs(LeRobotDataset, dataset_cfg)

        if client_train_episodes:
            train_dataset = LeRobotDataset(
                repo_id,
                episodes=client_train_episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                **dataset_kwargs,
            )
        else:
            train_dataset = _EmptyDataset()

        if client_test_episodes:
            test_dataset = LeRobotDataset(
                repo_id,
                episodes=client_test_episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                **dataset_kwargs,
            )
        else:
            test_dataset = _EmptyDataset()

        self.trainset = _MappedLeRobotDataset(train_dataset)
        self.testset = _MappedLeRobotDataset(test_dataset)

        self.repo_id = repo_id
        self.client_id = resolved_client_id
        self.train_episodes = client_train_episodes
        self.test_episodes = client_test_episodes
        self.meta = metadata

        logging.info(
            "LeRobot datasource ready for client %s: train episodes=%s, test episodes=%s",
            resolved_client_id,
            len(client_train_episodes),
            len(client_test_episodes),
        )

    @staticmethod
    def input_shape():
        """Return shape hint from configured transform image size when available."""
        params_cfg = getattr(Config(), "parameters", None)
        transform_cfg = _to_plain_dict(getattr(params_cfg, "transforms", None))

        image_size = _parse_size(transform_cfg.get("image_size"))
        if isinstance(image_size, tuple):
            return (3, image_size[0], image_size[1])

        raise ValueError(
            "LeRobot datasource input shape is dataset-dependent. "
            'Set "parameters.transforms.image_size" '
            "for a static shape hint."
        )
