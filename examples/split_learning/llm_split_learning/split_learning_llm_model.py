"""
Obtain LLM models from Huggingface, specifically designed for split learning
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import (
    get_peft_model,
    LoraConfig,
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)
from plato.config import Config


def get_lora_model(model):
    """Apply LoRA optimization over the model"""
    lora_config = Config().parameters.lora
    model = get_peft_model(model, LoraConfig(**lora_config._asdict()))
    model.print_trainable_parameters()
    print(model)
    return model


class BaseModel(torch.nn.Module):
    """
    The basic model loading hugginface model used for the server model and the client model
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
        client_module = self.base_model
        for module_name in client_module_names[:-1]:
            client_module = getattr(client_module, module_name)
        setattr(client_module, client_module_names[-1], client_layers)
        self.base_model.lm_head = torch.nn.Identity()

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
        #   used for training.
        self.server_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )
        transformer_module = self.base_model
        for module_name in Config().parameters.model.transformer_module_name.split("."):
            transformer_module = getattr(transformer_module, module_name)
        server_layers = transformer_module[self.cut_layer :]
        server_module_names = Config().parameters.model.transformer_module_name.split(
            "."
        )
        server_module = self.server_model
        for module_name in server_module_names[:-1]:
            server_module = getattr(server_module, module_name)
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
        base_model_weights = self.base_model.state_dict()
        server_model_weights = self.server_model.state_dict()
        transformer_module = self.base_model
        for module_name in Config().parameters.model.transformer_module_name.split("."):
            transformer_module = getattr(transformer_module, module_name)
        layer_names = [
            basic_name + "." + str(index)
            for index in range(
                self.cut_layer,
                len(transformer_module),
            )
        ]
        for weight_name in base_model_weights.keys():
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
        self.base_model.load_state_dict(base_model_weights)

    def forward_from(self, inputs, labels):
        """
        Forward from the intermediate feature on the server.
        """
        labels = labels.long()
        outputs = self.server_model(inputs_embeds=inputs, labels=labels)
        return outputs
