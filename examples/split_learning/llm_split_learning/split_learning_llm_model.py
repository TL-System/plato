"""
Obtain LLM models from HuggingFace, specifically designed for split learning
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from plato.config import Config


def get_lora_model(model):
    """Apply LoRA optimization over the model"""
    lora_config = Config().parameters.lora
    model = get_peft_model(model, LoraConfig(**lora_config._asdict()))
    model.print_trainable_parameters()
    return model


def get_module(start_module: torch.nn.Module, module_names):
    """
    Recursively get a PyTorch module starting from the start module with
    a given list of module names.
    """
    module = start_module
    for module_name in module_names:
        module = getattr(module, module_name)
    return module


class BaseModel(torch.nn.Module):
    """
    The basic model loading HuggingFace model used for the server model and the client model
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = Config().trainer.model_name
        use_auth_token = None
        if hasattr(Config().parameters, "huggingface_token"):
            use_auth_token = Config().parameters.huggingface_token
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": use_auth_token,
        }

        self.config = AutoConfig.from_pretrained(self.model_name, **config_kwargs)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            cache_dir=Config().params["model_path"] + "/huggingface",
            token=use_auth_token,
        )
        self.cut_layer = Config().parameters.model.cut_layer

    def get_input_embeddings(self):
        """
        Return the base model get input embeddings.
        """
        return self.base_model.get_input_embeddings()

    def forward(self, inputs):
        """
        The forward function for the base model.
        """
        return self.base_model(inputs)


class ClientModel(BaseModel):
    """
    The model on the clients in split learning with LLM.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # replace the layers in the base model
        # which should be on the cloud with Identity layers()
        transformer_module = self.base_model
        for module_name in Config().parameters.model.transformer_module_name.split("."):
            transformer_module = getattr(transformer_module, module_name)
        client_layers = transformer_module[: self.cut_layer]
        client_module_names = Config().parameters.model.transformer_module_name.split(
            "."
        )
        client_module = get_module(self.base_model, client_module_names[:-1])
        setattr(client_module, client_module_names[-1], client_layers)
        # Set layers not on the clients to Identity
        for layer in Config().parameters.model.layers_after_transformer:
            layer = layer.split(".")
            if len(layer) > 1:
                module = get_module(self.base_model, layer[:-1])
                setattr(module, layer[-1], torch.nn.Identity())
            else:
                setattr(self.base_model, layer[0], torch.nn.Identity())
        # Apply LoRA optimization
        if hasattr(Config().parameters, "lora"):
            self.base_model = get_lora_model(self.base_model)

    def forward(self, inputs):
        """
        The forward function on the client
        """
        inputs = inputs.long()
        return self.base_model(input_ids=inputs, return_dict=False)

    def forward_to(self, inputs):
        """
        Forward to the cut layer and output intermediate feature
        """
        outputs = self.forward(inputs)
        return outputs[0]


class ServerModel(BaseModel):
    """
    The model used on the cloud
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # In this design, we have two copies of the model
        # The first copy of the model is the whole model which is used for test.
        # The second copy of the model only contains the layers on the server
        # used for training.
        self.server_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )
        transformer_module = get_module(
            self.base_model,
            Config().parameters.model.transformer_module_name.split("."),
        )
        server_layers = transformer_module[self.cut_layer :]
        server_module_names = Config().parameters.model.transformer_module_name.split(
            "."
        )
        server_module = get_module(self.server_model, server_module_names[:-1])
        setattr(server_module, server_module_names[-1], server_layers)
        # Apply LoRA optimization
        if hasattr(Config().parameters, "lora"):
            self.base_model = get_lora_model(self.base_model)
            self.server_model = get_lora_model(self.server_model)

    def copy_weight(self):
        """
        Copy the weights of the training model to the testing model
        """
        basic_name = Config().parameters.model.transformer_module_name
        # There will a module named base_model.model in LoRA model
        if hasattr(Config().parameters, "lora"):
            basic_name = "base_model.model." + basic_name
        base_model_weights = self.base_model.state_dict()
        server_model_weights = self.server_model.state_dict()

        transformer_module = self.base_model
        for module_name in basic_name.split("."):
            transformer_module = getattr(transformer_module, module_name)
        layer_names = [
            basic_name + "." + str(index)
            for index in range(
                self.cut_layer,
                len(transformer_module),
            )
        ]
        for weight_name in base_model_weights.keys():
            # Copy the weights of transformer blocks
            for layer_index, layer_name in enumerate(layer_names):
                if layer_name in weight_name:
                    suffix = weight_name[
                        weight_name.find(layer_name) + len(layer_name) :
                    ]
                    # The name should be completely matched
                    if not suffix[0] == ".":
                        continue
                    server_weight_name = basic_name + "." + str(layer_index) + suffix
                    base_model_weights[weight_name] = server_model_weights[
                        server_weight_name
                    ]
            # Copy the weights of layers after transformer blocks
            for layer in Config().parameters.model.layers_after_transformer:
                layer_name = basic_name + "." + layer
                if layer_name in weight_name:
                    base_model_weights[weight_name] = server_model_weights[weight_name]

        self.base_model.load_state_dict(base_model_weights)

    def forward_from(self, inputs, labels):
        """
        Forward from the intermediate feature on the server.
        """
        labels = labels.long()
        outputs = self.server_model(inputs_embeds=inputs, labels=labels)
        return outputs
