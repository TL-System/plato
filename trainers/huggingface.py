"""
The Transformer models from HuggingFace for natural language processing.
"""

from config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.base import Model
from trainers import basic


class Trainer(basic.Trainer):
    """The trainer for HuggingFace transformer models for natural language processing. """
    def __init__(self, model: Model, client_id=0):
        super().__init__(model, client_id)
        transformer_type = Config().trainer.model
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_type),
        self.model = AutoModelForCausalLM.from_pretrained(transformer_type)
