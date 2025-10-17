"""Entities needed to conduct federated learning with LoRA adapters."""

import logging
from typing import Dict, Iterable, Tuple

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaTokenizer,
)

from plato.algorithms import fedavg
from plato.config import Config
from plato.datasources import base
from plato.models import registry as model_registry
from plato.trainers import huggingface as hf_trainer


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


class LoraCollateWrapper:
    """Wraps the HuggingFace data collator to integrate with Plato's strategies."""

    def __init__(self, tokenizer):
        self.collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def __call__(
        self, examples: Iterable[Dict]
    ) -> Tuple[hf_trainer.HuggingFaceBatch, torch.Tensor]:
        """Collate raw examples into batched tensors."""
        batch = self.collator(examples)
        labels = batch.pop("labels")
        return hf_trainer.HuggingFaceBatch(batch), labels


class Trainer(hf_trainer.Trainer):
    """Trainer leveraging the shared HuggingFace implementation for LoRA fine-tuning."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)

        # Ensure padding configuration expected by causal language modeling.
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        collate_wrapper = LoraCollateWrapper(self.tokenizer)
        self._collate_wrapper = collate_wrapper

        if hasattr(self.data_loader_strategy, "collate_fn"):
            self.data_loader_strategy.collate_fn = collate_wrapper

        if hasattr(self.testing_strategy, "collate_fn"):
            self.testing_strategy.collate_fn = collate_wrapper


class DataSource(base.DataSource):
    """A datasource with custom training and validation datasets for LoRA fine-tuning."""

    def __init__(self):
        super().__init__()

        dataset_name = Config().data.dataset_name
        logging.info("Dataset: %s", dataset_name)

        dataset = load_dataset(dataset_name)

        column_names = dataset["train"].column_names

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
                return_attention_mask=True,
            )

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

        train_data = tokenized_datasets["train"].shuffle(seed=42)
        val_data = tokenized_datasets["validation"].shuffle(seed=42)

        self.trainset = train_data
        self.testset = val_data


class Algorithm(fedavg.Algorithm):
    def extract_weights(self, model=None):
        # Extract LoRA weights
        return {
            k: v.cpu()
            for k, v in get_peft_model_state_dict(self.model.base_model).items()
        }

    def load_weights(self, weights):
        # Load LoRA weights
        return set_peft_model_state_dict(self.model.base_model, weights)
