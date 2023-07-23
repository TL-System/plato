"""Entities needed to conduct federated learning with LoRA adapters using Plato."""

import logging
from functools import partial
import math

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from plato.algorithms import fedavg
from plato.datasources import base
from plato.trainers import huggingface
from plato.config import Config
from plato.models import registry as model_registry
from peft import (
    get_peft_model,
    LoraConfig,
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)


class LoraModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        base_model = model_registry.get()
        lora_config = Config().parameters.lora
        self.base_model = get_peft_model(
            base_model, LoraConfig(**lora_config._asdict())
        )
        self.base_model.print_trainable_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


class Trainer(huggingface.Trainer):
    """A custom trainer with custom training and testing loops."""

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The training loop for HuggingFace models.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        """

        self.training_args.num_train_epochs = config["epochs"]
        self.training_args.per_device_train_batch_size = config["batch_size"]
        self.training_args.max_steps = 4

        self.trainer = huggingface.SampledHuggingFaceTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=trainset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer,
                mlm=False,
            ),
            sampler=sampler,
            callbacks=self.trainer_callbacks,
        )

        self.trainer.train()

    def test_model(
        self, config, testset, sampler=None, **kwargs
    ):  # pylint: disable=unused-argument
        """The testing loop for HuggingFace models.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        self.training_args.per_device_eval_batch_size = config["batch_size"]

        self.trainer = huggingface.SampledHuggingFaceTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=testset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer,
                mlm=False,
            ),
            sampler=sampler,
            callbacks=None,
        )

        metrics = self.trainer.evaluate()

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        return perplexity


class DataSource(base.DataSource):
    """A custom datasource with custom training and validation datasets."""

    def __init__(self):
        super().__init__()

        # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model_name = Config().trainer.model_name
        if "llama" in model_name:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        def tokenize_function(examples):
            return tokenizer(
                examples["review"],
                max_length=128,
                padding="max_length",
                truncation=True,
            )

        dataset = load_dataset("srivatsavaasista/textgenerator-ds-mini")

        tokenized_datasets = dataset.map(tokenize_function)

        train_data = tokenized_datasets["train"].shuffle(seed=42)
        val_data = tokenized_datasets["validation"].shuffle(seed=42)

        self.trainset = train_data
        self.testset = val_data


class Algorithm(fedavg.Algorithm):
    def extract_weights(self, model=None):
        # Extract LoRA wegiths
        return {
            k: v.cpu()
            for k, v in get_peft_model_state_dict(self.model.base_model).items()
        }

    def load_weights(self, weights):
        # Load LoRA weights
        return set_peft_model_state_dict(self.model.base_model, weights)