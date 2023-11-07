"""
Obtain LLM models from Huggingface, specifically designed for split learning
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from plato.config import Config


class BaseModel(torch.nn.Module):
    """
    The basic model loading hugginface model used for the server model and the client model
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = Config().trainer.model_name
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }

        self.config = AutoConfig.from_pretrained(self.model_name, **config_kwargs)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            cache_dir=Config().params["model_path"] + "/huggingface",
        )
        self.cut_layer = Config().trainer.cut_layer_index

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
        self.base_model.transformer.h = self.base_model.transformer.h[: self.cut_layer]
        self.base_model.transformer.ln_f = torch.nn.Identity()
        self.base_model.lm_head = torch.nn.Identity()

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
        self.server_model.transformer.h = self.base_model.transformer.h[
            self.cut_layer :
        ]

    def copy_weight_from_training_model_to_testing_model(self):
        """
        Copy the weights of the training model to the testing model
        """
        basic_name = "transformer.h."
        base_model_weights = self.base_model.state_dict()
        server_model_weights = self.server_model.state_dict()
        layers_name = [
            basic_name + str(index)
            for index in range(self.cut_layer, len(self.base_model.transformer.h))
        ]
        for weight_name in base_model_weights.keys():
            for layer_index, layer_name in enumerate(layers_name):
                if layer_name in weight_name:
                    suffix = weight_name[
                        weight_name.find(layer_name) + len(layer_name) :
                    ]
                    server_weight_name = basic_name + str(layer_index) + suffix
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
