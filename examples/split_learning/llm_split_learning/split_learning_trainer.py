"""
A split learning trainer for large language model fine-tuning.

This trainer uses the composable trainer architecture with custom strategies
and callbacks to handle HuggingFace transformers in a split learning setting.
"""

from collections import OrderedDict
from typing import Optional

import evaluate
from torch import Tensor, reshape
from torch.utils.data import RandomSampler, Sampler
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
)
from transformers import Trainer as HuggingFaceTrainer

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers import split_learning
from plato.trainers.strategies.base import TestingStrategy, TrainingContext

# ============================================================================
# Helper Functions
# ============================================================================


def preprocess_logits_for_metrics(logits, labels):
    """Preprocess the logits for calculating accuracy."""
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    """Calculate the accuracy for evaluation stage."""
    metric = evaluate.load("accuracy")
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    return metric.compute(predictions=preds, references=labels)


# ============================================================================
# Custom HuggingFace Trainer with Sampling Support
# ============================================================================


class SampledHuggingFaceTrainer(HuggingFaceTrainer):
    """
    Training and testing loops for HuggingFace's transformer models for natural
    language processing with custom sampler support.
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        sampler,
        callbacks,
    ):
        """Initialize the HuggingFace trainer with custom sampler."""
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=callbacks,
        )
        self.sampler = sampler

    def _get_train_sampler(self) -> Optional[Sampler]:
        """Get training sampler."""
        if self.sampler is None:
            return RandomSampler(self.train_dataset)
        return self.sampler

    def _get_eval_sampler(self, eval_dataset) -> Optional[Sampler]:
        """Get evaluation sampler."""
        if self.sampler is None:
            return super()._get_eval_sampler(eval_dataset)
        return self.sampler


# ============================================================================
# Custom Testing Strategy for LLM Split Learning
# ============================================================================


class LLMSplitLearningTestingStrategy(TestingStrategy):
    """
    Testing strategy for LLM split learning using HuggingFace models.

    This strategy uses HuggingFace's evaluation pipeline for testing.
    """

    def __init__(self, tokenizer, training_args):
        """Initialize with tokenizer and training arguments."""
        self.tokenizer = tokenizer
        self.training_args = training_args

    def test_model(self, model, config, testset, sampler, context):
        """
        Test the model using HuggingFace's evaluation pipeline.

        Arguments:
            model: The model to test
            config: Testing configuration
            testset: Test dataset
            sampler: Optional data sampler
            context: Training context

        Returns:
            Test accuracy as float
        """
        batch_size = config.get("batch_size", 32)
        self.training_args.per_device_eval_batch_size = batch_size

        # Copy weights for split learning models
        if hasattr(model, "copy_weight"):
            model.copy_weight()

        # Get base model if available
        base_model = model.base_model if hasattr(model, "base_model") else model

        tester = SampledHuggingFaceTrainer(
            model=base_model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=testset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            sampler=sampler,
            callbacks=None,
        )

        metrics = tester.evaluate()

        # Save other metric information such as accuracy
        tester.log_metrics("eval", metrics)
        return metrics["eval_accuracy"]


# ============================================================================
# Custom Callbacks for LLM Split Learning
# ============================================================================


class LLMTokenizerCallback(TrainerCallback):
    """
    Callback to initialize and manage tokenizer for LLM training.

    Handles tokenizer initialization and embedding resizing.
    """

    def __init__(self):
        """Initialize the callback."""
        self.tokenizer = None

    def on_trainer_initialized(self, trainer, **kwargs):
        """Initialize tokenizer and resize embeddings if needed."""
        tokenizer_kwargs = {
            "cache_dir": Config().params["data_path"],
            "use_fast": True,
            "revision": "main",
            "use_auth_token": None,
        }
        model_name = Config().trainer.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        # Resize embeddings to avoid index errors
        embedding_size = trainer.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            trainer.model.resize_token_embeddings(len(self.tokenizer))

        # Store tokenizer in trainer for easy access
        trainer.tokenizer = self.tokenizer


class LLMTrainingArgsCallback(TrainerCallback):
    """
    Callback to initialize HuggingFace training arguments.
    """

    def __init__(self):
        """Initialize the callback."""
        self.training_args = None

    def on_trainer_initialized(self, trainer, **kwargs):
        """Initialize HuggingFace training arguments."""
        parser = HfArgumentParser(TrainingArguments)

        (self.training_args,) = parser.parse_args_into_dataclasses(
            args=[
                "--output_dir=" + Config.params["checkpoint_path"],
                "--report_to=none",
            ]
        )

        # Store training args in trainer for easy access
        trainer.training_args = self.training_args


# ============================================================================
# LLM Split Learning Trainer
# ============================================================================


class Trainer(split_learning.Trainer):
    """
    The split learning trainer to fine-tune LLM.

    This trainer extends the base split learning trainer with HuggingFace-specific
    functionality via custom callbacks and testing strategy. It handles tokenization,
    embedding resizing, and uses HuggingFace's evaluation pipeline for testing.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the LLM split learning trainer.

        Arguments:
            model: The model to train (HuggingFace model)
            callbacks: List of callback classes or instances
        """
        # Create LLM-specific callbacks
        tokenizer_callback = LLMTokenizerCallback()
        training_args_callback = LLMTrainingArgsCallback()

        # Combine with provided callbacks
        all_callbacks = [tokenizer_callback, training_args_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent split learning trainer
        super().__init__(model=model, callbacks=all_callbacks)

        # Replace testing strategy with LLM-specific one
        # Note: We need to do this after parent init so tokenizer and training_args are set
        self.testing_strategy = LLMSplitLearningTestingStrategy(
            self.tokenizer, self.training_args
        )

    def process_training_samples_before_retrieving(self, training_samples):
        """
        Process training samples before retrieving in split learning.

        Reshapes inputs and labels for split learning format.

        Arguments:
            training_samples: Dictionary containing input_ids and labels

        Returns:
            Tuple of (inputs, labels) as tensors
        """
        inputs = training_samples["input_ids"]
        labels = training_samples["labels"]

        # Convert to list format
        for index, input_item in enumerate(inputs):
            inputs[index] = input_item.tolist()
        inputs = Tensor(inputs)
        inputs = reshape(inputs, (inputs.shape[1], inputs.shape[0]))

        for index, label_item in enumerate(labels):
            labels[index] = label_item.tolist()
        labels = Tensor(labels)
        labels = reshape(labels, (labels.shape[1], labels.shape[0]))

        return (inputs, labels)

    def update_weights_before_cut(
        self, current_weights: OrderedDict, weights: OrderedDict
    ):
        """
        Update weights before the cut layer in split learning.

        Arguments:
            current_weights: Current model weights
            weights: New weights to apply

        Returns:
            Updated weights dictionary
        """
        for client_layer_name, client_layer_parameters in weights.items():
            current_weights[client_layer_name] = client_layer_parameters
        return current_weights

    def server_forward_from(self, batch, config):
        """
        Server-side forward pass from the cut layer in split learning.

        Arguments:
            batch: Batch of data (inputs, labels)
            config: Training configuration

        Returns:
            Tuple of (loss, gradients, batch_size)
        """
        inputs, labels = batch
        batch_size = inputs.size(0)
        inputs = inputs.detach().requires_grad_(True)
        outputs = self.model.forward_from(inputs, labels)
        loss = outputs.loss
        loss.backward()
        grad = inputs.grad
        return loss, grad, batch_size
