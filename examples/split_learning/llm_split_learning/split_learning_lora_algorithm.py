"""
A split learning algorithm supporting LoRA fine-tuning LLMs.
"""

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from plato.algorithms import split_learning


class Algorithm(split_learning.Algorithm):
    """
    Extract and load only the LoRA weights.
    """

    def extract_weights(self, model=None):
        # Extract LoRA wegiths
        return {
            k: v.cpu()
            for k, v in get_peft_model_state_dict(self.model.base_model).items()
        }

    def load_weights(self, weights):
        # Load LoRA weights
        return set_peft_model_state_dict(self.model.base_model, weights)
