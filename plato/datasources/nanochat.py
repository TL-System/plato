"""
Streaming datasource backed by the vendored Nanochat project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Nanochat datasource requires PyTorch. "
        "Install torch via the project's optional dependencies."
    ) from exc

from plato.config import Config
from plato.datasources.base import DataSource as BaseDataSource
from plato.utils.third_party import ThirdPartyImportError, ensure_nanochat_importable

DEFAULT_VOCAB_SIZE = 50304
DEFAULT_SEQUENCE_LENGTH = 2048


def _resolve_base_dir(base_dir: str | Path | None) -> Path | None:
    if base_dir is None:
        return None
    return Path(base_dir).expanduser().resolve()


def _parquet_available(base_dir: Path | None) -> bool:
    try:
        ensure_nanochat_importable()
        from nanochat.dataset import (
            DATA_DIR,
            list_parquet_files,
        )
    except (ThirdPartyImportError, ImportError):  # pragma: no cover - defensive
        return False

    if base_dir is not None:
        candidate_dir = base_dir / "base_data"
        if not candidate_dir.exists():
            return False
        parquet_dir = candidate_dir
    else:
        parquet_dir = Path(DATA_DIR)

    try:
        return len(list_parquet_files(str(parquet_dir))) > 0
    except FileNotFoundError:
        return False


@dataclass
class _SyntheticState:
    generator: torch.Generator


class NanochatStreamingDataset(IterableDataset):
    """Iterable dataset yielding (inputs, targets) token tensors."""

    def __init__(
        self,
        *,
        split: str,
        batch_size: int,
        sequence_length: int,
        mode: str,
        base_dir: Path | None,
        max_batches: int | None,
        tokenizer_threads: int,
        tokenizer_batch_size: int,
        device: str,
        vocab_size: int,
        synthetic_seed: int,
    ):
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")

        if mode not in {"auto", "parquet", "synthetic"}:
            raise ValueError("mode must be 'auto', 'parquet', or 'synthetic'.")

        self.split = split
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.base_dir = base_dir
        self.max_batches = max_batches
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer_batch_size = tokenizer_batch_size
        self.device = device
        self.vocab_size = vocab_size
        self.synthetic_seed = synthetic_seed

        resolved_mode = mode
        if resolved_mode == "auto":
            resolved_mode = "parquet" if _parquet_available(base_dir) else "synthetic"
        self.mode = resolved_mode
        self._synthetic_state: _SyntheticState | None = None

        # Configure Nanochat's base directory if provided.
        if self.base_dir is not None:
            os.environ.setdefault("NANOCHAT_BASE_DIR", str(self.base_dir))

    def _synthetic_iterable(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        if self._synthetic_state is None:
            generator = torch.Generator()
            generator.manual_seed(self.synthetic_seed)
            self._synthetic_state = _SyntheticState(generator=generator)

        generator = self._synthetic_state.generator
        while True:
            tokens = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(self.batch_size, self.sequence_length + 1),
                dtype=torch.long,
                generator=generator,
            )
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            yield inputs, targets

    def _parquet_iterable(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        ensure_nanochat_importable()
        from nanochat.dataloader import (
            tokenizing_distributed_data_loader,
        )

        loader = tokenizing_distributed_data_loader(
            self.batch_size,
            self.sequence_length,
            split=self.split,
            tokenizer_threads=self.tokenizer_threads,
            tokenizer_batch_size=self.tokenizer_batch_size,
            device=self.device,
        )
        for inputs, targets in loader:
            yield (
                inputs.to(dtype=torch.long).contiguous(),
                targets.to(dtype=torch.long).contiguous(),
            )

    def __iter__(self):
        iterable = (
            self._parquet_iterable()
            if self.mode == "parquet"
            else self._synthetic_iterable()
        )

        for batch_index, batch in enumerate(iterable):
            if self.max_batches is not None and batch_index >= self.max_batches:
                break
            yield batch

    def __len__(self) -> int:
        if self.max_batches is None:
            raise TypeError("Streaming dataset does not have a finite length.")
        return self.max_batches


class DataSource(BaseDataSource):
    """Plato datasource exposing Nanochat token streams."""

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        mode: str = "auto",
        base_dir: str | Path | None = None,
        max_train_batches: int | None = 64,
        max_val_batches: int | None = 8,
        tokenizer_threads: int = 4,
        tokenizer_batch_size: int = 128,
        device: str = "cpu",
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        synthetic_seed: int = 42,
    ):
        super().__init__()

        cfg_data = getattr(Config(), "data", None)
        if cfg_data is not None:
            mode = getattr(cfg_data, "mode", mode)
            base_dir = getattr(cfg_data, "base_dir", base_dir)
            max_train_batches = getattr(
                cfg_data, "max_train_batches", max_train_batches
            )
            max_val_batches = getattr(cfg_data, "max_val_batches", max_val_batches)
            tokenizer_threads = getattr(
                cfg_data, "tokenizer_threads", tokenizer_threads
            )
            tokenizer_batch_size = getattr(
                cfg_data, "tokenizer_batch_size", tokenizer_batch_size
            )
            device = getattr(cfg_data, "device", device)
            vocab_size = getattr(cfg_data, "vocab_size", vocab_size)
            synthetic_seed = getattr(cfg_data, "synthetic_seed", synthetic_seed)

        config = getattr(Config(), "parameters", None)
        model_conf = getattr(config, "model", None)
        default_seq_len = DEFAULT_SEQUENCE_LENGTH
        if model_conf is not None and hasattr(model_conf, "_asdict"):
            seq_len_candidate = model_conf._asdict().get("sequence_len")
            if isinstance(seq_len_candidate, int) and seq_len_candidate > 0:
                default_seq_len = seq_len_candidate

        resolved_sequence_len = sequence_length or default_seq_len
        resolved_batch_size = batch_size or getattr(
            getattr(Config(), "trainer", None), "batch_size", 1
        )

        resolved_base_dir = _resolve_base_dir(base_dir)
        dataset_mode = mode
        if dataset_mode == "auto":
            dataset_mode = (
                "parquet" if _parquet_available(resolved_base_dir) else "synthetic"
            )

        self.trainset: NanochatStreamingDataset = NanochatStreamingDataset(
            split="train",
            batch_size=resolved_batch_size,
            sequence_length=resolved_sequence_len,
            mode=dataset_mode,
            base_dir=resolved_base_dir,
            max_batches=max_train_batches,
            tokenizer_threads=tokenizer_threads,
            tokenizer_batch_size=tokenizer_batch_size,
            device=device,
            vocab_size=vocab_size,
            synthetic_seed=synthetic_seed,
        )
        self.testset: NanochatStreamingDataset = NanochatStreamingDataset(
            split="val",
            batch_size=resolved_batch_size,
            sequence_length=resolved_sequence_len,
            mode=dataset_mode,
            base_dir=resolved_base_dir,
            max_batches=max_val_batches,
            tokenizer_threads=tokenizer_threads,
            tokenizer_batch_size=tokenizer_batch_size,
            device=device,
            vocab_size=vocab_size,
            synthetic_seed=synthetic_seed + 1,
        )
        self.sequence_length = resolved_sequence_len
        self.batch_size = resolved_batch_size
        self.mode = dataset_mode

    @staticmethod
    def input_shape():
        """Return the default input shape (sequence length)."""
        return (DEFAULT_SEQUENCE_LENGTH,)

    def num_train_examples(self) -> int:
        dataset = self.trainset
        if dataset.max_batches is None:
            raise RuntimeError(
                "Nanochat datasource streams infinity; configure max_train_batches to report size."
            )
        return dataset.max_batches * dataset.batch_size

    def num_test_examples(self) -> int:
        dataset = self.testset
        if dataset.max_batches is None:
            raise RuntimeError(
                "Nanochat datasource streams infinity; configure max_val_batches to report size."
            )
        return dataset.max_batches * dataset.batch_size
