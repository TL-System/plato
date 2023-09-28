"""
Implementation for text-to-image with textual inversion.
"""
import itertools
import logging

from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


# In Textual-Inversion we only train the newly added embedding vector,
# so lets freeze rest of the model parameters here
def freeze_params(params):
    for param in params:
        param.requires_grad = False


class GenerationPromptLearner(nn.Module):
    """A lightweight network."""

    def __init__(self, **kwargs):
        """Define the model."""
        super().__init__()
        # self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
        model_type, model_name = GenerationPromptLearner.get_pretrained_config_info(
            **kwargs
        )
        pretrained_model_name_or_path = f"{model_type}/{model_name}"

        # home_dir = os.path.expanduser("~")
        # catch_dir = ".cache/huggingface/hub/models--stabilityai--stable-diffusion-2"
        # pretrained_model_name_or_path = os.path.join(home_dir, catch_dir)

        # Load the tokenizer and add the placeholder token as a additional special token.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        # `placeholder_token` will be added in the tokenizer so we resize the token embeddings here,
        # this will a new embedding vector in the token embeddings for our `placeholder_token`
        self.text_encoder.resize_token_embeddings(len(self.tokenizer) + 1)

        # setting the placeholder to be the default one
        self.placeholder_token = ""
        self.placeholder_token_id = len(self.tokenizer)

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)

    def tokenizer_add_placeholder(self, placeholder_token: str):
        """Adding the placeholader to current tokenizer.
        a unknown place_holder will be added to the end of current tokenzier,
        original length: 49408
        after adding place_holder: 49409
        thus, placeholder_token_id: 49408
        """
        self.placeholder_token = placeholder_token
        num_added_tokens = self.tokenizer.add_tokens(placeholder_token)

        # equals to: self.placeholder_token_id = len(self.tokenizer) - 1
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            placeholder_token
        )
        logging.info(
            "Added #%d tokens with placeholder %s and id %s ",
            num_added_tokens,
            placeholder_token,
            self.placeholder_token_id,
        )
        return num_added_tokens

    def initial_placeholder_embed(self, initializer_token):
        """Initializing the embed of placeholder with the embed of an existed token id."""
        # Get token ids for our placeholder and initializer token. This code block will complain if initializer string is not a single token
        # Convert the initializer_token, placeholder_token to ids
        token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[self.placeholder_token_id] = token_embeds[initializer_token_id]

    def set_placeholder_embedding(self, embedding):
        """Setting the embedding for the placeholder."""
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        embedding = embedding.to(token_embeds.device)
        logging.info(
            "Setting embedding of placeholder %s with id %d",
            self.placeholder_token,
            self.placeholder_token_id,
        )
        token_embeds[self.placeholder_token_id] = embedding

    def get_placeholder_embedding(self):
        """Setting the embedding for the placeholder.

        Remember to use .clone() to make the obtained embedding of
        placeholder has its own physical memory, otherwise, the memory
        will be the whole embeddings.
        """
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        logging.info(
            "Getting embedding of placeholder %s with id %d",
            self.placeholder_token,
            self.placeholder_token_id,
        )
        return token_embeds[self.placeholder_token_id].clone()

    def forward(self, input_ids):
        """Forwarding the encoder to get the hidden status.

        :param input_ids: A `torch.Tensor` with shape [B, 77]
         where the 77 is the context length from a pre-defined tokenizer.

        :return A `torch.Tensor` with shape [B, 77, encoding_length]
        """
        # Get the text embedding for conditioning
        return self.text_encoder(input_ids)[0]

    def load_state_dict(self, state_dict, strict):
        """Assigning the embedding to placeholder position."""
        self.set_placeholder_embedding(state_dict[self.placeholder_token])

    def state_dict(self, **kwargs):
        """Getting the prompt embeddings."""
        return {self.placeholder_token: self.get_placeholder_embedding()}

    @staticmethod
    def get_pretrained_config_info(**kwargs):
        """Getting the pre-trained information of config."""
        # stabilityai
        model_type = kwargs["model_type"] if "model_type" in kwargs else "stabilityai"

        # for example, stable-diffusion-2
        model_name = (
            kwargs["model_name"] if "model_name" in kwargs else "stable-diffusion-2"
        )

        return model_type, model_name
