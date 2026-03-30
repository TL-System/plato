"""
A data source for the HuggingFace datasets.

For more information about the HuggingFace datasets, refer to:

https://huggingface.co/docs/datasets/quicktour.html
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Mapping
from typing import Any, cast

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    testing_utils,
)
from transformers.utils import logging as hf_logging

from plato.config import Config
from plato.datasources import base


def _sanitize_cache_component(value: Any) -> str:
    """Return a filesystem-friendly cache path component."""
    if value is None:
        return "none"
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return normalized or "default"


def _dataset_cache_path(
    data_path: str,
    *,
    dataset_name: str,
    dataset_config: Any,
    preprocessing_mode: str,
    train_split: str,
    validation_split: str,
) -> str:
    """Build a stable cache path for the raw downloaded dataset."""
    signature = "|".join(
        [
            str(dataset_name),
            str(dataset_config),
            preprocessing_mode,
            train_split,
            validation_split,
        ]
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    prefix = "__".join(
        [
            _sanitize_cache_component(dataset_name),
            _sanitize_cache_component(dataset_config),
            _sanitize_cache_component(preprocessing_mode),
            _sanitize_cache_component(train_split),
            _sanitize_cache_component(validation_split),
        ]
    )
    return os.path.join(data_path, f"{prefix}__{digest}")


def _resolve_split_name(
    dataset: Mapping[str, Any], preferred: str, *, fallback: str | None = None
) -> str:
    """Resolve a split name with an optional fallback when absent."""
    if preferred in dataset:
        return preferred
    if fallback is not None and fallback in dataset:
        return fallback
    available = ", ".join(sorted(str(name) for name in dataset.keys()))
    raise KeyError(
        f"Dataset split '{preferred}' not available. Available splits: {available}."
    )


class DataSource(base.DataSource):
    """A data source for HuggingFace datasets supporting multiple preprocessing modes."""

    def __init__(self, **kwargs):
        super().__init__()

        data_cfg = Config().data
        dataset_name = data_cfg.dataset_name
        dataset_config = getattr(data_cfg, "dataset_config", None)
        train_split_name = getattr(data_cfg, "train_split", "train")
        requested_validation_split = getattr(data_cfg, "validation_split", "validation")
        preprocessing_mode = getattr(
            data_cfg,
            "preprocessing_mode",
            getattr(data_cfg, "format", "corpus_lm"),
        )

        logging.info("Dataset: %s", dataset_name)

        saved_data_path = _dataset_cache_path(
            Config().params["data_path"],
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            preprocessing_mode=preprocessing_mode,
            train_split=train_split_name,
            validation_split=requested_validation_split,
        )

        if os.path.exists(saved_data_path):
            self.dataset = load_from_disk(saved_data_path)
        else:
            dataset_kwargs: dict[str, Any] = {}
            if dataset_config is not None:
                dataset_kwargs["name"] = dataset_config
            self.dataset = load_dataset(dataset_name, **dataset_kwargs)
            save_to_disk = getattr(self.dataset, "save_to_disk", None)
            if callable(save_to_disk):
                save_to_disk(saved_data_path)

        parser = HfArgumentParser(cast(Any, TrainingArguments))
        (self.training_args,) = parser.parse_args_into_dataclasses(
            args=["--output_dir=/tmp", "--report_to=none"]
        )
        self.training_args = cast(TrainingArguments, self.training_args)

        tokenizer_name = getattr(Config().trainer, "tokenizer_name", None)
        model_name = (
            tokenizer_name
            if isinstance(tokenizer_name, str) and tokenizer_name
            else Config().trainer.model_name
        )
        auth_token = getattr(getattr(Config(), "parameters", None), "huggingface_token", None)
        config_kwargs = {
            "cache_dir": Config().params["model_path"],
            "revision": "main",
            "use_auth_token": auth_token,
        }
        tokenizer_kwargs = {
            "cache_dir": Config().params["data_path"],
            "use_fast": True,
            "revision": "main",
            "use_auth_token": auth_token,
        }

        self.config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=self.config, **tokenizer_kwargs
        )
        self.tok_logger = hf_logging.get_logger("transformers.tokenization_utils_base")

        self.preprocessing_mode = str(preprocessing_mode)
        self.text_column_name = getattr(data_cfg, "text_field", "text")
        self.column_names = [self.text_column_name]
        self.preprocess_num_proc = getattr(data_cfg, "preprocessing_num_proc", 4)
        self.block_size = getattr(data_cfg, "block_size", 128)
        self.saved_data_path = saved_data_path

        self.train_split_name = _resolve_split_name(self.dataset, train_split_name)
        self.validation_split_name = _resolve_split_name(
            self.dataset,
            requested_validation_split,
            fallback="test" if requested_validation_split == "validation" else None,
        )

        self.trainset = self.preprocess_split(self.dataset[self.train_split_name])
        self.testset = self.preprocess_split(self.dataset[self.validation_split_name])

    def num_train_examples(self):
        return len(self.require_trainset())

    def num_test_examples(self):
        return len(self.require_testset())

    def get_train_set(self):
        return self.require_trainset()

    def get_test_set(self):
        return self.require_testset()

    @staticmethod
    def input_shape():
        """Returns the input shape of the dataset, useful for building
        a TF model."""
        raise ValueError("Not implemented.")

    def tokenize_function(self, examples):
        """Using the tokenizer from AutoTokenizer to tokenize the text."""
        with testing_utils.CaptureLogger(self.tok_logger) as cl:
            output = self.tokenizer(examples[self.text_column_name])
        if "Token indices sequence length is longer than the" in cl.out:
            self.tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be "
                "chunked into smaller bits before being passed to the model."
            )
        return output

    def group_texts(self, examples):
        """Concatenate texts then split them into language-modeling blocks."""
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size

        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess_split(self, dataset_split):
        """Dispatch preprocessing according to the configured mode."""
        if self.preprocessing_mode == "corpus_lm":
            return self.preprocess_corpus_lm(dataset_split)
        if self.preprocessing_mode == "chat_sft":
            return self.preprocess_chat_sft(dataset_split)
        raise ValueError(
            f"Unsupported HuggingFace preprocessing mode: {self.preprocessing_mode}"
        )

    def preprocess_corpus_lm(self, dataset_split):
        """Tokenize and group a plain-text corpus for causal language modeling."""
        training_args = cast(TrainingArguments, self.training_args)
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = dataset_split.map(
                self.tokenize_function,
                batched=True,
                num_proc=self.preprocess_num_proc,
                remove_columns=self.column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )

        configured_block_size = getattr(Config().data, "block_size", None)
        block_size = configured_block_size or self.tokenizer.model_max_length
        if block_size > 1024:
            logging.warning(
                "The tokenizer picked seems to have a very large `model_max_length` "
                "%s. Picking 1024 instead.",
                self.tokenizer.model_max_length,
            )
            block_size = 1024
        self.block_size = int(block_size)

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                self.group_texts,
                batched=True,
                num_proc=self.preprocess_num_proc,
                load_from_cache_file=True,
                desc=f"Grouping texts in chunks of {self.block_size}",
            )

        return lm_datasets

    def preprocess_chat_sft(self, dataset_split):
        """Placeholder for follow-up chat-SFT preprocessing support."""
        raise NotImplementedError(
            "chat_sft preprocessing is implemented in a follow-up issue."
        )

    def preprocess_data(self, datasets):
        """Backward-compatible alias for the legacy corpus LM preprocessing path."""
        return self.preprocess_corpus_lm(datasets)
