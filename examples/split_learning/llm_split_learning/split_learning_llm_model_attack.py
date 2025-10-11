"""
An LLM model on the server which has certain hack functions helping recover the
    private data on the clients
"""

import logging

import torch
from split_learning_llm_model import (
    ServerModel as ServerModelHonest,
)
from split_learning_llm_model import (
    get_lora_model,
    get_module,
)
from transformers import AutoModelForCausalLM

from plato.config import Config


class ServerModelCurious(ServerModelHonest):
    """
    The server model has an estimated client model
        with guessed parameters by the server.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.guessed_client_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )
        # Construct a guessed client model with layers should be one the clients
        transformer_module = self.guessed_client_model
        for module_name in Config().parameters.model.transformer_module_name.split("."):
            transformer_module = getattr(transformer_module, module_name)
        client_layers = transformer_module[: self.cut_layer]
        client_module_names = Config().parameters.model.transformer_module_name.split(
            "."
        )
        client_module = get_module(self.guessed_client_model, client_module_names[:-1])
        setattr(client_module, client_module_names[-1], client_layers)
        # Set layers not on the clients to Identity
        for layer in Config().parameters.model.layers_after_transformer:
            layer = layer.split(".")
            if len(layer) > 1:
                module = get_module(self.guessed_client_model, layer[:-1])
                setattr(module, layer[-1], torch.nn.Identity())
            else:
                setattr(self.guessed_client_model, layer[0], torch.nn.Identity())
        self.guessed_client_model.lm_head = torch.nn.Identity()
        # Apply LoRA optimization
        if hasattr(Config().parameters, "lora"):
            self.guessed_client_model = get_lora_model(self.guessed_client_model)

    def calibrate_guessed_client(self, calibrate=True):
        """
        Calibrate the weights of the guessed client model to the weights of the client model,
            if the client sent the model to the server for testing.
        We just need to do this before the server is ready to launch the attack.
        """
        if calibrate:
            base_model_weight = self.base_model.state_dict()
            guessed_client_model_weights = self.guessed_client_model.state_dict()
            for weight_name in guessed_client_model_weights.keys():
                if not isinstance(
                    guessed_client_model_weights[weight_name],
                    torch.nn.Identity,
                ):
                    guessed_client_model_weights[weight_name] = base_model_weight[
                        weight_name
                    ]
            self.guessed_client_model.load_state_dict(guessed_client_model_weights)
        else:
            logging.info(
                "In current attacks, the guessed client model weights are not calibrated"
            )
