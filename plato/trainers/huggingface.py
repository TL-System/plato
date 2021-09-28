"""
Training and testing loops for HuggingFace's transformer models for natural
language processing.
"""
import torch

from transformers import AutoTokenizer
from transformers import Trainer as HuggingFaceTrainer
from transformers import TrainingArguments

from plato.config import Config
from plato.trainers import basic


class SampledHuggingFaceTrainer(HuggingFaceTrainer):
    """
    Training and testing loops for HuggingFace's transformer models for natural
    language processing.
    """
    def __init__(self, model, args, train_dataset, eval_dataset, sampler):
        super().__init__(model=model,
                         args=args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset)
        self.sampler = sampler
        self.trainset = train_dataset
        self.batch_size = Config().trainer.batch_size

    def get_train_dataloader(self):
        if self.sampler is not None:
            return torch.utils.data.DataLoader(dataset=self.trainset,
                                               shuffle=False,
                                               batch_size=self.batch_size,
                                               sampler=self.sampler.get())


class Trainer(basic.Trainer):
    """The trainer for HuggingFace transformer models for natural language processing. """
    def __init__(self, model=None):
        super().__init__(model)

        self.trainer = None

        model_checkpoint = Config().trainer.model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                       use_fast=True)

        self.model.train()

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    @staticmethod
    def group_texts(examples):
        """Concatenate all texts. """
        block_size = 128

        concatenated_examples = {
            k: sum(examples[k], [])
            for k in examples.keys()
        }

        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, we could add padding if the model supported it
        # instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess_data(self, datasets):
        tokenized_datasets = datasets.map(self.tokenize_function,
                                          batched=True,
                                          num_proc=4,
                                          remove_columns=["text"])

        return tokenized_datasets.map(
            Trainer.group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )

    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument
        """The training loop for HuggingFace models.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """
        training_args = TrainingArguments(
            "test-clm",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
        )

        print("Training dataset preprocessed.")

        self.trainer = SampledHuggingFaceTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.preprocess_data(trainset),
            eval_dataset=None,
            sampler=sampler)

        self.trainer.train()

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        """The testing loop for HuggingFace models.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        self.trainer = SampledHuggingFaceTrainer(
            model=self.model,
            args=None,
            train_dataset=None,
            eval_dataset=self.preprocess_data(testset),
            sampler=None)

        return self.trainer.evaluate()
