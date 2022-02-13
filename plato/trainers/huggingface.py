"""
Training and testing loops for HuggingFace's transformer models for natural
language processing.
"""
import math
from typing import Optional

from torch.utils.data import RandomSampler, Sampler

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from transformers import Trainer as HuggingFaceTrainer
from transformers import TrainingArguments, default_data_collator

from plato.config import Config
from plato.trainers import basic


class SampledHuggingFaceTrainer(HuggingFaceTrainer):
    """
    Training and testing loops for HuggingFace's transformer models for natural
    language processing.
    """

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer,
                 data_collator, sampler):
        super().__init__(model=model,
                         args=args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         data_collator=data_collator)
        self.sampler = sampler

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.sampler is None:
            return RandomSampler(self.train_dataset)

        return self.sampler


class Trainer(basic.Trainer):
    """The trainer for HuggingFace transformer models for natural language processing. """

    def __init__(self, model=None):
        super().__init__(model)

        self.trainer = None
        self.model.train()

        parser = HfArgumentParser(TrainingArguments)
        self.training_args, = parser.parse_args_into_dataclasses(
            args=['--output_dir=/tmp', '--report_to=none'])

        model_checkpoint = Config().trainer.model_checkpoint
        config_kwargs = {
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token": None,
        }
        self.config = AutoConfig.from_pretrained(model_checkpoint,
                                                 **config_kwargs)

        tokenizer_kwargs = {
            "cache_dir": None,
            "use_fast": True,
            "revision": 'main',
            "use_auth_token": None,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                       config=self.config,
                                                       **tokenizer_kwargs)

    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument
        """The training loop for HuggingFace models.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """

        self.training_args.num_train_epoches = config['epochs']
        self.training_args.per_device_train_batch_size = config['batch_size']

        self.trainer = SampledHuggingFaceTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=trainset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            sampler=sampler)

        self.trainer.train()

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        """The testing loop for HuggingFace models.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        self.trainer = HuggingFaceTrainer(model=self.model,
                                          args=self.training_args,
                                          train_dataset=None,
                                          eval_dataset=testset,
                                          tokenizer=self.tokenizer,
                                          data_collator=default_data_collator)

        metrics = self.trainer.evaluate()

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        return perplexity
