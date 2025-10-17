"""Entities needed to conduct federated learning with LoRA adapters."""

import logging
import math
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.utils.data
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
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import CustomCollateFnDataLoaderStrategy
from plato.trainers.strategies.base import (
    TrainingContext,
    TrainingStepStrategy,
    TestingStrategy,
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


class HuggingFaceBatch(dict):
    """A dictionary-like batch returned by HuggingFace collators."""

    def to(self, device):
        """Move all tensor values to the given device."""
        for key, value in self.items():
            if hasattr(value, "to"):
                self[key] = value.to(device)
        return self


class LoraCollateWrapper:
    """Wraps the HuggingFace data collator to integrate with Plato's strategies."""

    def __init__(self, tokenizer):
        self.collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def __call__(self, examples: Iterable[Dict]) -> Tuple[HuggingFaceBatch, torch.Tensor]:
        """Collate raw examples into batched tensors."""
        batch = self.collator(examples)
        labels = batch.pop("labels")
        return HuggingFaceBatch(batch), labels


class LoraTrainingStepStrategy(TrainingStepStrategy):
    """Training step that leverages HuggingFace causal language modeling."""

    def training_step(
        self,
        model,
        optimizer,
        examples,
        labels,
        loss_criterion,  # pylint: disable=unused-argument
        context: TrainingContext,
    ):
        """Execute a HuggingFace forward/backward pass."""
        optimizer.zero_grad()

        batch_inputs = dict(examples)
        batch_inputs["labels"] = labels

        outputs = model(**batch_inputs)
        loss = getattr(outputs, "loss", outputs[0])
        loss.backward()
        optimizer.step()

        return loss.detach()


class LoraTestingStrategy(TestingStrategy):
    """Testing strategy returning perplexity for language modeling."""

    def __init__(self, collate_fn: LoraCollateWrapper):
        self.collate_fn = collate_fn

    def test_model(self, model, config, testset, sampler, context: TrainingContext):
        """Evaluate model perplexity on the provided dataset."""
        batch_size = config.get("batch_size", 1)

        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                sampler_obj = sampler.get()
            else:
                sampler_obj = sampler
        else:
            sampler_obj = None

        data_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_obj,
            collate_fn=self.collate_fn,
        )

        model.to(context.device)
        model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch_inputs, labels in data_loader:
                batch_inputs = batch_inputs.to(context.device)
                labels = labels.to(context.device)
                batch_inputs["labels"] = labels

                outputs = model(**batch_inputs)
                loss = getattr(outputs, "loss", outputs[0])

                valid_tokens = (labels != -100).sum().item()
                total_tokens += max(valid_tokens, 1)
                total_loss += loss.item() * max(valid_tokens, 1)

        model.train()

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens

        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")

        return perplexity


class Trainer(ComposableTrainer):
    """A trainer using the composable strategy API for LoRA fine-tuning."""

    def __init__(self, model=None, callbacks=None):
        model_name = Config().trainer.model_name

        if "llama" in model_name:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        collate_wrapper = LoraCollateWrapper(tokenizer)

        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=None,
            optimizer_strategy=None,
            training_step_strategy=LoraTrainingStepStrategy(),
            lr_scheduler_strategy=None,
            model_update_strategy=None,
            data_loader_strategy=CustomCollateFnDataLoaderStrategy(
                collate_fn=collate_wrapper,
                num_workers=0,
                pin_memory=True,
            ),
            testing_strategy=LoraTestingStrategy(collate_wrapper),
        )

        self.tokenizer = tokenizer

        # Ensure model checkpoints can be saved when HuggingFace names contain slashes.
        params = Config().params
        try:
            model_path = params["model_path"]
        except (TypeError, KeyError):
            model_path = None
        model_name = Config().trainer.model_name
        sub_dir = os.path.dirname(model_name)
        if model_path and sub_dir:
            os.makedirs(os.path.join(model_path, sub_dir), exist_ok=True)


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
        # Extract LoRA wegiths
        return {
            k: v.cpu()
            for k, v in get_peft_model_state_dict(self.model.base_model).items()
        }

    def load_weights(self, weights):
        # Load LoRA weights
        return set_peft_model_state_dict(self.model.base_model, weights)
