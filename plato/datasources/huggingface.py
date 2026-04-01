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


def _legacy_dataset_cache_path(
    data_path: str, *, dataset_name: str, dataset_config: Any
) -> str:
    """Return the historical raw-dataset cache path for backward compatibility."""
    return f"{data_path}/{dataset_name}_{dataset_config}"


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


def _coerce_token_sequence(value: Any, *, field_name: str) -> list[int]:
    """Normalize tokenizer outputs into a flat list of token ids."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise TypeError(
                f"Expected a single token sequence for '{field_name}', got {len(value)} sequences."
            )
        value = value[0]

    if not isinstance(value, list):
        raise TypeError(
            f"Expected '{field_name}' to be a token sequence, got {type(value)}."
        )

    return [int(token) for token in value]


def _normalize_chat_template_output(
    rendered: Any,
) -> tuple[list[int], list[int] | None]:
    """Handle tokenizers that return either raw ids or a mapping payload."""
    attention_mask = None
    input_ids_source = rendered

    if isinstance(rendered, Mapping):
        if "input_ids" not in rendered:
            raise KeyError(
                "Chat template tokenization output must include 'input_ids'."
            )
        input_ids_source = rendered["input_ids"]
        if "attention_mask" in rendered:
            attention_mask = _coerce_token_sequence(
                rendered["attention_mask"],
                field_name="attention_mask",
            )

    input_ids = _coerce_token_sequence(input_ids_source, field_name="input_ids")

    if attention_mask is not None and len(attention_mask) != len(input_ids):
        raise ValueError(
            "Chat template output produced mismatched input_ids and attention_mask lengths."
        )

    return input_ids, attention_mask


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
        legacy_saved_data_path = _legacy_dataset_cache_path(
            Config().params["data_path"],
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        )

        if os.path.exists(saved_data_path):
            self.dataset = load_from_disk(saved_data_path)
        elif os.path.exists(legacy_saved_data_path):
            self.dataset = load_from_disk(legacy_saved_data_path)
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
        self.messages_field = getattr(data_cfg, "messages_field", "messages")
        self.label_strategy = getattr(data_cfg, "label_strategy", "assistant_only")
        self.max_seq_length = getattr(data_cfg, "max_seq_length", None)
        self.pack_sequences = getattr(data_cfg, "pack_sequences", False)
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
        block_size = configured_block_size if configured_block_size is not None else self.block_size
        block_size = int(block_size)
        if block_size > 1024:
            logging.warning(
                "The configured block size %s is very large. Picking 1024 instead.",
                block_size,
            )
            block_size = 1024
        self.block_size = block_size

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                self.group_texts,
                batched=True,
                num_proc=self.preprocess_num_proc,
                load_from_cache_file=True,
                desc=f"Grouping texts in chunks of {self.block_size}",
            )

        return lm_datasets

    def _chat_max_seq_length(self) -> int:
        """Resolve the effective max sequence length for chat SFT examples."""
        configured = self.max_seq_length
        if configured is not None:
            return int(configured)

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", 1024)
        if not isinstance(tokenizer_limit, int) or tokenizer_limit <= 0:
            return 1024
        return min(tokenizer_limit, 1024)

    def _build_chat_labels(
        self, messages: list[dict[str, Any]], input_ids: list[int]
    ) -> list[int]:
        """Build labels for chat SFT according to the configured label strategy."""
        if self.label_strategy == "full_sequence":
            return list(input_ids)
        if self.label_strategy != "assistant_only":
            raise ValueError(
                f"Unsupported chat label strategy: {self.label_strategy}"
            )

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise AttributeError(
                "Tokenizer must expose apply_chat_template() for chat_sft preprocessing."
            )

        labels = [-100] * len(input_ids)
        previous_length = 0
        for index, message in enumerate(messages):
            rendered = self.tokenizer.apply_chat_template(
                messages[: index + 1],
                tokenize=True,
                add_generation_prompt=False,
            )
            rendered_ids, _ = _normalize_chat_template_output(rendered)
            current_length = min(len(rendered_ids), len(input_ids))
            if message.get("role") == "assistant":
                labels[previous_length:current_length] = input_ids[
                    previous_length:current_length
                ]
            previous_length = current_length
        return labels

    def preprocess_chat_sft(self, dataset_split):
        """Tokenize chat-style supervision examples one conversation at a time."""
        if self.pack_sequences:
            raise NotImplementedError(
                "chat_sft preprocessing does not support pack_sequences=true."
            )
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise AttributeError(
                "Tokenizer must expose apply_chat_template() for chat_sft preprocessing."
            )

        column_names_raw = getattr(dataset_split, "column_names", None)
        if not isinstance(column_names_raw, list):
            raise AttributeError("Chat split must expose 'column_names'.")
        remove_columns = [str(name) for name in column_names_raw]
        max_seq_length = self._chat_max_seq_length()
        messages_field = self.messages_field

        def tokenize_chat_example(example: dict[str, Any]) -> dict[str, list[int]]:
            messages = example[messages_field]
            if not isinstance(messages, list):
                raise TypeError(
                    f"Expected a list of messages under '{messages_field}'."
                )

            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            input_ids, attention_mask = _normalize_chat_template_output(rendered)
            labels = self._build_chat_labels(messages, input_ids)

            if len(input_ids) > max_seq_length:
                input_ids = input_ids[-max_seq_length:]
                labels = labels[-max_seq_length:]
                if attention_mask is not None:
                    attention_mask = attention_mask[-max_seq_length:]

            if attention_mask is None:
                attention_mask = [1] * len(input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        training_args = cast(TrainingArguments, self.training_args)
        with training_args.main_process_first(desc="chat sft preprocessing"):
            tokenized = dataset_split.map(
                tokenize_chat_example,
                num_proc=None,
                remove_columns=remove_columns,
                load_from_cache_file=True,
                desc="Running chat template preprocessing",
            )

        return tokenized

    def preprocess_data(self, datasets):
        """Backward-compatible alias for the legacy corpus LM preprocessing path."""
        return self.preprocess_corpus_lm(datasets)
