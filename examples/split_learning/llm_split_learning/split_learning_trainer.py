"""A split learning trainer for large language model fine-tuning"""
from typing import Optional
from collections import OrderedDict

from torch.utils.data import RandomSampler, Sampler
from torch import Tensor, reshape
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
)
from transformers import Trainer as HuggingFaceTrainer
import evaluate

from plato.trainers import split_learning

from plato.config import Config


# pylint:disable=unused-argument
def preprocess_logits_for_metrics(logits, labels):
    "Preprocess the logits for calculating accuracy"
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    """Calculate the accuracy for evaluation stage"""
    metric = evaluate.load("accuracy")
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    return metric.compute(predictions=preds, references=labels)


class SampledHuggingFaceTrainer(HuggingFaceTrainer):
    """
    Training and testing loops for HuggingFace's transformer models for natural
    language processing.
    """

    # pylint:disable=too-many-arguments
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
        if self.sampler is None:
            return RandomSampler(self.train_dataset)

        return self.sampler

    def _get_eval_sampler(self, eval_dataset) -> Optional[Sampler]:
        if self.sampler is None:
            return super()._get_eval_sampler(eval_dataset)

        return self.sampler


class Trainer(split_learning.Trainer):
    """The split learning trainer to fine-tune LLM."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        # We resize the embeddings to avoid index errors.
        tokenizer_kwargs = {
            "cache_dir": Config().params["data_path"],
            "use_fast": True,
            "revision": "main",
            "use_auth_token": None,
        }
        model_name = Config().trainer.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        # self.training args for HuggingFace training
        parser = HfArgumentParser(TrainingArguments)

        (self.training_args,) = parser.parse_args_into_dataclasses(
            args=[
                "--output_dir=" + Config.params["checkpoint_path"],
                "--report_to=none",
            ]
        )

    # Redesign the evaluation stage.
    def test_model_split_learning(self, batch_size, testset, sampler=None):
        """The testing loop for HuggingFace models.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        self.training_args.per_device_eval_batch_size = batch_size
        self.model.copy_weight()

        tester = SampledHuggingFaceTrainer(
            model=self.model.base_model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=testset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            sampler=sampler,
            callbacks=None,
        )

        metrics = tester.evaluate()

        # save other metric information such as accuracy
        tester.log_metrics("eval", metrics)
        return metrics["eval_accuracy"]

    # Redesign the training stage specific to Split Learning.
    def process_training_samples_before_retrieving(self, training_samples):
        inputs = training_samples["input_ids"]
        labels = training_samples["labels"]
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
        for client_layer_name, client_layer_parameters in weights.items():
            current_weights[client_layer_name] = client_layer_parameters
        return current_weights

    def server_forward_from(self, batch, config):
        inputs, labels = batch
        batch_size = inputs.size(0)
        inputs = inputs.detach().requires_grad_(True)
        outputs = self.model.forward_from(inputs, labels)
        loss = outputs.loss
        loss.backward()
        grad = inputs.grad
        return loss, grad, batch_size
