"""
The Transformer models from HuggingFace for natural language processing.
"""

from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer as HuggingFaceTrainer

from config import Config
from trainers import basic


class Trainer(basic.Trainer):
    """The trainer for HuggingFace transformer models for natural language processing. """
    def __init__(self, client_id=0):
        super().__init__(client_id)

        self.trainer = None

        model_checkpoint = Config().trainer.model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                       use_fast=True)

        self.model.train()

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    @staticmethod
    def group_texts(examples):
        # Concatenate all texts.
        block_size = 128

        concatenated_examples = {
            k: sum(examples[k], [])
            for k in examples.keys()
        }

        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
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

    def train_model(self, config, trainset, cut_layer=None):  # pylint: disable=unused-argument
        """The training loop for YOLOv5.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """
        training_args = TrainingArguments(
            "test-clm",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
        )

        lm_datasets = self.preprocess_data(trainset)

        print("Training dataset preproessed.")

        self.trainer = HuggingFaceTrainer(
            model=self.model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
        )

        self.trainer.train()

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        """The testing loop for YOLOv5.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        return self.trainer.evaluate()
