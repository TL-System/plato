"""
Implementation of the trainer for Federated Textual Inversion.

"""
import os
import logging
from typing import Dict
import glob

import torch
from plato.config import Config
from accelerate import Accelerator
import torch.nn.functional as F

import fedti_stable_diffusion
from plato.trainers import basic
from text_inversion_dataset import TextualInversionDataset


class Trainer(basic.Trainer):
    """A trainer to perform learning of Text-to-Image model."""

    def __init__(self, model=None, callbacks=None):
        """Define the prompts of the trainer."""

        super().__init__(model, callbacks)

        # the model maintained only by the client
        self.personalized_model = None
        self.personalized_model_name: str = None
        # what is it that you are teaching?
        # `object` enables you to teach the model a new object to be used,
        # `style` allows you to teach the model a new style one can use.
        self.what_to_teach = ""  # @param ["object", "style"]

        # Settings for your newly created concept
        self.concept_name: str = None

        # `placeholder_token` is the token you are going to use to represent your new concept
        # (so when you prompt the model, you will say "A `` in an amusement park").
        # We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
        self.placeholder_token: str = None
        # `initializer_token` is a word that can summarise what your new concept is,
        #  to be used as a starting point
        self.initializer_token = ""  # @param {type:"string"}

        # accelerator
        self.accelerator = None

        # for prompts for text-to-image
        # the encodings should be the dict in which
        # the key is the str presenting the text.
        self.samples_prompts_encodings: Dict[str, torch.Tensor] = {}

        # prompts encodigns of the client,
        # a Tensor with shape, [n_prompts, n_encodings]
        self.local_prompts_encodings: Dict[str, torch.Tensor] = {}

        # prompts encodings from the server
        self.global_prompts_encodings: Dict[str, torch.Tensor] = {}

        # repeat the source images
        self.repeats = None
        self.learning_rate = None
        self.gradient_accumulation_steps = None
        self.train_batch_size = None

    def create_what_to_teach(self):
        """Getting what_to_teach."""
        self.what_to_teach = Config().algorithm.what_to_teach

    def create_concept(self):
        """Getting the concept for the client."""
        # set constant concept
        self.concept_name = Config().algorithm.concept_name

    def create_placeholder_token(self):
        """Setting the placeholder token."""
        self.placeholder_token = f"<{self.concept_name}>"

    def create_initializer_token(self):
        """Getting the initial token assigned to placeholder token
        as the starting point."""
        self.initializer_token = Config().algorithm.initializer_token

    def define_personalized_model(self):
        """Define the personalized model to this trainer."""

        model_type = Config().algorithm.personalization.model_type
        model_name = Config().algorithm.personalization.model_name

        self.personalized_model = fedti_stable_diffusion.Text2ImageSDPipeline(
            model_type=model_type, model_name=model_name
        )
        self.personalized_model_name = model_name
        logging.info(
            "[Client #%d] Defined the model %s from %s",
            self.client_id,
            self.personalized_model_name,
            model_type,
        )

    def train_run_start(self, config):
        """Before running, convert the config to be ones for personalization."""

        batch_size = config["batch_size"]
        self.gradient_accumulation_steps = config["gradient_accumulation_steps"]
        self.repeats = config["repeats"]
        gradient_checkpointing = config["gradient_checkpointing"]

        self.learning_rate = config["learning_rate"]

        self.model.to(self.device)
        self.personalized_model.to(self.device)
        self.model.train()
        self.personalized_model.eval()

        if self.accelerator is None:
            if torch.cuda.is_available():
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    mixed_precision=config["mixed_precision"],
                )
            else:
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    cpu=True,
                )
        if gradient_checkpointing:
            self.model.text_encoder.gradient_checkpointing_enable()
            self.personalized_model.unet.enable_gradient_checkpointing()

        self.personalized_model.prepare_status(accelerator=self.accelerator)
        self.train_batch_size = batch_size

    def get_data_path(self, clien_id, trainset):
        """Get the checkpoint path for current client."""
        data_path = Config.params["data_path"]
        data_folder_name = os.path.basename(trainset.get_base_folder())

        return os.path.join(
            data_path, data_folder_name, "Clients", f"client_{clien_id}"
        )

    # pylint: disable=unused-argument
    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Getting train loader of text inversion dataset."""

        client_data_root = self.get_data_path(self.client_id, trainset)

        n_images = (
            len(glob.glob(client_data_root + "/*.jpeg"))
            + len(glob.glob(client_data_root + "/*.jpg"))
            if os.path.exists(client_data_root)
            else 0
        )

        if not os.path.exists(client_data_root) or n_images == 0:
            collate_func = lambda batch: batch
            basic_loader = torch.utils.data.DataLoader(
                dataset=trainset,
                shuffle=False,
                batch_size=1,
                sampler=sampler,
                collate_fn=collate_func,
            )
            os.makedirs(client_data_root, exist_ok=False)
            for _, batch in enumerate(basic_loader):
                (images, _, images_filename) = batch[0]
                images.save(os.path.join(client_data_root, images_filename))

        # Create the Dataset and Dataloader
        train_dataset = TextualInversionDataset(
            data_root=client_data_root,
            tokenizer=self.model.tokenizer,
            size=self.personalized_model.vae.config.sample_size,
            placeholder_token=self.placeholder_token,
            repeats=self.repeats,
            learnable_property=self.what_to_teach,  # Option selected above between object and style
            center_crop=False,
            set="train",
        )
        n_source_images = TextualInversionDataset.count_images(client_data_root)
        logging.info("[Client #%d] source images: %s", self.client_id, n_source_images)
        logging.info(
            "[Client #%d] train images: %s", self.client_id, len(train_dataset)
        )

        return torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

    def get_optimizer(self, model):
        """Returns the optimizer."""
        # Initialize the optimizer
        # only optimize the embeddings
        learning_rate = (
            self.learning_rate
            * self.gradient_accumulation_steps
            * self.train_batch_size
            * self.accelerator.num_processes
        )
        optimizer = torch.optim.AdamW(
            model.text_encoder.get_input_embeddings().parameters(),
            lr=learning_rate,
        )

        model, optimizer, self.train_loader = self.accelerator.prepare(
            model, optimizer, self.train_loader
        )

        return optimizer

    def get_lr_scheduler(self, config, optimizer):
        """Do not define lr_scheduler."""
        return None

    def get_loss_criterion(self):
        """Do not define loss criterion."""

        def compute_loss(noise_pred, target, **kwargs):
            return (
                F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
            )

        return compute_loss

    def extract_placeholder_hidden_states(self, input_ids, hidden_states):
        """Extracting the hidden states for placeholder."""
        target_encodings = []
        placeholder_token_id = self.model.placeholder_token_id
        for row_idx, row in enumerate(input_ids):
            target_indices = (row == placeholder_token_id).nonzero(as_tuple=True)[0]
            target_encoding = hidden_states[row_idx, target_indices, :]
            target_encodings.append(target_encoding)

        target_encodings = torch.cat(target_encodings, dim=0)
        return target_encodings

    def compute_local_soft_prompts(self):
        """Getting local soft prompts."""
        if self.samples_prompts_encodings is not None:
            # encodings: [n_samples, n_encodings]
            for prompt, encodings in self.samples_prompts_encodings.items():
                # [n_encodings]
                self.local_prompts_encodings[prompt] = torch.mean(encodings, dim=0)

    def train_epoch_start(self, config):
        """Resetting the samples prompts encodings."""
        super().train_epoch_start(config)

        self.samples_prompts_encodings = {}
        self.model.train()

    def train_step_end(self, config, batch=None, loss=None):
        """Waiting for all processes after one iteration."""
        self.accelerator.wait_for_everyone()

    def collect_samples_prompt_encodings(self, prompts, prompts_encodings):
        """Collecting samples prompt encoding."""

        for prompt, encodings in zip(prompts, prompts_encodings):
            encodings = encodings.unsqueeze(0).detach().cpu()
            if prompt in self.samples_prompts_encodings:
                self.samples_prompts_encodings[prompt] = torch.cat(
                    (self.samples_prompts_encodings[prompt], encodings),
                    dim=0,
                )
            else:
                self.samples_prompts_encodings[prompt] = encodings

    def perform_forward_and_backward_passes(self, config, examples, labels):
        self.optimizer.zero_grad()
        with self.accelerator.accumulate(self.model):
            # Get the text embedding for conditioning
            input_ids = examples["input_ids"]
            encoder_hidden_states = self.model(input_ids)

            # [batch_size, n_encodings]
            prompts_encodings = self.extract_placeholder_hidden_states(
                input_ids, hidden_states=encoder_hidden_states
            )
            self.collect_samples_prompt_encodings(examples["text"], prompts_encodings)

            # Get diffusion outputs
            noise_pred, target = self.personalized_model(
                pixel_values=examples["pixel_values"],
                hidden_states=encoder_hidden_states,
            )

            batch_prompts = examples["text"]
            loss = self._loss_criterion(
                noise_pred,
                target,
                batch_prompts=batch_prompts,
                batch_prompts_encoding=prompts_encodings,
            )
            self._loss_tracker.update(loss, target.size(0))

            if "create_graph" in config:
                self.accelerator.backward(loss, create_graph=config["create_graph"])
            else:
                self.accelerator.backward(loss)

            # Zero out the gradients for all token embeddings except the newly added
            # embeddings for the concept, as we only want to optimize the concept embeddings
            if self.accelerator.num_processes > 1:
                grads = (
                    self.model.text_encoder.module.get_input_embeddings().weight.grad
                )
            else:
                grads = self.model.text_encoder.get_input_embeddings().weight.grad
            # Get the index for tokens that we want to zero the grads for
            index_grads_to_zero = (
                torch.arange(len(self.model.tokenizer))
                != self.model.placeholder_token_id
            )
            grads.data[index_grads_to_zero, :] = grads.data[
                index_grads_to_zero, :
            ].fill_(0)

        self.optimizer.step()

        return loss

    def set_init_prompts(self, prompts):
        """Setting prompts for the trainer."""
        self.initial_prompts = None
