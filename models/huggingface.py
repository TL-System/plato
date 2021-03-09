"""
The Transformer models from HuggingFace for natural language processing.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM


class Model:
    """The HuggingFace Transformer models for natural language processing. """
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    @staticmethod
    def is_valid_model_type(model_type):
        return model_type.startswith('huggingface')

    @staticmethod
    def get_model_from_type(model_type):
        """
        Obtaining an instance of a HuggingFace Transformer model provided
        that the name is valid.
        """
        if not Model.is_valid_model_type(model_type):
            raise ValueError('Invalid model type: {}'.format(model_type))

        transformer_type = model_type.split('_')[1]

        return Model(
            tokenizer=AutoTokenizer.from_pretrained(transformer_type),
            model=AutoModelForCausalLM.from_pretrained(transformer_type))
