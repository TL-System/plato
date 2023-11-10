"""
A data source for the HuggingFace datasets.

For more information about the HuggingFace datasets, refer to:

https://huggingface.co/docs/datasets/quicktour.html
"""

import logging
import os

from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from transformers import TrainingArguments, testing_utils, utils

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """A data source for the HuggingFace datasets."""

    def __init__(self, **kwargs):
        super().__init__()

        dataset_name = Config().data.dataset_name
        logging.info("Dataset: %s", dataset_name)

        if hasattr(Config.data, "dataset_config"):
            dataset_config = Config().data.dataset_config
        else:
            dataset_config = None

        saved_data_path = (
            f"{Config().params['data_path']}/{dataset_name}_{dataset_config}"
        )

        if os.path.exists(saved_data_path):
            # If the dataset has already been downloaded and saved
            self.dataset = load_from_disk(saved_data_path)
        else:
            # Download and save the dataset
            self.dataset = load_dataset(dataset_name, dataset_config)
            self.dataset.save_to_disk(saved_data_path)

        parser = HfArgumentParser(TrainingArguments)
        (self.training_args,) = parser.parse_args_into_dataclasses(
            args=["--output_dir=/tmp", "--report_to=none"]
        )

        model_name = Config().trainer.model_name
        use_auth_token = None
        if hasattr(Config().parameters, "huggingface_token"):
            use_auth_token = Config().parameters.huggingface_token
        config_kwargs = {
            "cache_dir": Config().params["model_path"],
            "revision": "main",
            "use_auth_token": use_auth_token,
        }
        tokenizer_kwargs = {
            "cache_dir": Config().params["data_path"],
            "use_fast": True,
            "revision": "main",
            "use_auth_token": use_auth_token,
        }

        self.config = AutoConfig.from_pretrained(model_name, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=self.config, **tokenizer_kwargs
        )
        self.tok_logger = utils.logging.get_logger(
            "transformers.tokenization_utils_base"
        )

        self.block_size = 128

        self.column_names = ["text"]
        self.text_column_name = "text"
        self.trainset = self.preprocess_data(self.dataset["train"])
        self.testset = self.preprocess_data(self.dataset["validation"])

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)

    def get_train_set(self):
        return self.trainset

    def get_test_set(self):
        return self.testset

    @staticmethod
    def input_shape():
        """Returns the input shape of the dataset, useful for building
        a TF model."""
        raise ValueError("Not implemented.")

    def tokenize_function(self, examples):
        """Using the tokenizer from AutoTokenizer to tokenize the text."""
        with testing_utils.CaptureLogger(self.tok_logger) as cl:
            output = self.tokenizer(examples[self.text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            self.tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be "
                "chunked into smaller bits before being passed to the model."
            )
        return output

    def group_texts(self, examples):
        """Concatenate all texts."""
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, we could add padding if the model supported it
        # instead of this drop, you can customize this part to your needs.
        total_length = (total_length // self.block_size) * self.block_size

        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess_data(self, datasets):
        """Tokenizing and grouping the raw dataset."""
        with self.training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = datasets.map(
                self.tokenize_function,
                batched=True,
                num_proc=4,
                remove_columns=self.column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )

        block_size = self.tokenizer.model_max_length
        if block_size > 1024:
            logging.warning(
                "The tokenizer picked seems to have a very large `model_max_length` "
                "%s. Picking 1024 instead.",
                self.tokenizer.model_max_length,
            )
            block_size = 1024

        with self.training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                self.group_texts,
                batched=True,
                num_proc=4,
                load_from_cache_file=True,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        return lm_datasets
