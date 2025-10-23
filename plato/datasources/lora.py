"""
LoRA-friendly datasource built on HuggingFace datasets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """
    A datasource that tokenizes HuggingFace text datasets for LoRA fine-tuning.

    Configuration (data section):
        dataset_name: Name on HuggingFace hub.
        dataset_config: Optional dataset subset/config name.
        train_split: Dataset split for training (default ``"train"``).
        validation_split: Dataset split for validation (default ``"validation"``).
        text_field: Field containing raw text (default ``"text"``).
        max_length: Token sequence length (default 128).
        shuffle_seed: Seed for deterministic shuffling (default 42).
    """

    def __init__(self, **kwargs):
        super().__init__()

        data_cfg = Config().data
        dataset_name = data_cfg.dataset_name
        dataset_config = getattr(data_cfg, "dataset_config", None)
        train_split = getattr(data_cfg, "train_split", "train")
        val_split = getattr(data_cfg, "validation_split", "validation")
        text_field = getattr(data_cfg, "text_field", "text")
        max_length = getattr(data_cfg, "max_length", 128)
        shuffle_seed = getattr(data_cfg, "shuffle_seed", 42)

        logging.info("Dataset: %s", dataset_name)

        dataset_kwargs: dict[str, Any] = {}
        if dataset_config is not None:
            dataset_kwargs["name"] = dataset_config

        dataset = load_dataset(dataset_name, **dataset_kwargs)

        train_split_dataset = dataset[train_split]
        if not isinstance(train_split_dataset, Dataset):
            raise TypeError(
                f"Split '{train_split}' is not a HuggingFace Dataset instance."
            )
        column_names_raw = getattr(train_split_dataset, "column_names", None)
        if not isinstance(column_names_raw, list):
            raise AttributeError("Training split must expose 'column_names'.")
        column_names: list[str] = [str(name) for name in column_names_raw]

        model_name = Config().trainer.model_name
        if "llama" in model_name.lower():
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        def tokenize_function(examples: dict[str, list[str]]):
            return tokenizer(
                examples[text_field],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

        train_tokenized = tokenized_datasets[train_split]
        shuffle_fn = getattr(train_tokenized, "shuffle", None)
        if not callable(shuffle_fn):
            raise AttributeError("Training split does not support shuffling.")
        train_data = shuffle_fn(seed=shuffle_seed)
        val_data: Any | None = None
        if val_split in tokenized_datasets:
            val_tokenized = tokenized_datasets[val_split]
            shuffle_val = getattr(val_tokenized, "shuffle", None)
            if callable(shuffle_val):
                val_data = shuffle_val(seed=shuffle_seed)
            else:
                val_data = val_tokenized

        self.trainset = train_data
        self.testset = val_data

    def num_train_examples(self):
        return len(self.require_trainset())

    def num_test_examples(self):
        testset = self.testset
        return len(testset) if testset is not None else 0

    def get_train_set(self):
        return self.require_trainset()

    def get_test_set(self):
        return self.testset
