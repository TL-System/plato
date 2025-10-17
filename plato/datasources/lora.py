"""
LoRA-friendly datasource built on HuggingFace datasets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from datasets import load_dataset
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

        dataset_kwargs: Dict[str, Any] = {}
        if dataset_config is not None:
            dataset_kwargs["name"] = dataset_config

        dataset = load_dataset(dataset_name, **dataset_kwargs)

        column_names: List[str] = dataset[train_split].column_names

        model_name = Config().trainer.model_name
        if "llama" in model_name.lower():
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        def tokenize_function(examples: Dict[str, List[str]]):
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

        train_data = tokenized_datasets[train_split].shuffle(seed=shuffle_seed)
        val_data: Optional[Any] = None
        if val_split in tokenized_datasets:
            val_data = tokenized_datasets[val_split].shuffle(seed=shuffle_seed)

        self.trainset = train_data
        self.testset = val_data

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset) if self.testset is not None else 0

    def get_train_set(self):
        return self.trainset

    def get_test_set(self):
        return self.testset
