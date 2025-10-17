"""
Strategy-based trainer for HuggingFace transformer models.

This implementation uses Plato's composable trainer architecture by wiring
HuggingFace data handling through strategy objects instead of overriding
`load_model`/`save_model` hooks.

"""

import logging
import math
import os
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.data
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    LlamaTokenizer,
    TrainingArguments,
    default_data_collator,
)
from transformers import (
    TrainerCallback as HFTrainerCallback,
)
from transformers.trainer_callback import TrainerControl, TrainerState

from plato.callbacks.trainer import TrainerCallback as PlatoTrainerCallback
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import CustomCollateFnDataLoaderStrategy
from plato.trainers.strategies.base import (
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)


class HuggingFaceBatch(dict):
    """Dictionary-style batch that supports `.to(device)` like torch tensors."""

    def to(self, device):
        for key, value in self.items():
            if hasattr(value, "to"):
                self[key] = value.to(device)
        return self


class HuggingFaceCollateWrapper:
    """Wraps the default HuggingFace data collator for Plato data loader strategy."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(
        self, examples: Iterable[dict]
    ) -> Tuple[HuggingFaceBatch, Optional[torch.Tensor]]:
        batch = default_data_collator(examples)
        labels = batch.pop("labels", None)
        if labels is None:
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                labels = input_ids.clone()
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    labels = labels.masked_fill(attention_mask == 0, -100)
                elif (
                    self.tokenizer is not None
                    and self.tokenizer.pad_token_id is not None
                ):
                    labels = labels.masked_fill(
                        labels == self.tokenizer.pad_token_id, -100
                    )
        return HuggingFaceBatch(batch), labels


def _resolve_hf_loss(outputs, labels, *, allow_fallback: bool = True):
    """
    Resolve a loss tensor from HuggingFace model outputs.

    Args:
        outputs: HuggingFace model outputs (ModelOutput, tuple, tensor, etc.).
        labels: Labels tensor if available.
        allow_fallback: Whether to compute loss manually when not provided.

    Returns:
        A torch.Tensor representing the loss.

    Raises:
        ValueError: If no loss tensor can be determined.
    """
    loss = getattr(outputs, "loss", None)
    if loss is None:
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            loss = outputs[0]
        else:
            loss = outputs

    if torch.is_tensor(loss):
        return loss

    if not allow_fallback:
        raise ValueError("HuggingFace model did not return a tensor loss.")

    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, dict):
        logits = outputs.get("logits")
    if logits is None and isinstance(outputs, tuple) and len(outputs) > 0:
        logits = outputs[0]

    if logits is None or labels is None:
        logits_shape = None if logits is None else tuple(logits.shape)
        labels_shape = None if labels is None else tuple(labels.shape)
        logging.error(
            "Unable to resolve HuggingFace loss: logits=%s labels=%s outputs_type=%s",
            "None" if logits is None else f"{type(logits)} shape={logits_shape}",
            "None" if labels is None else f"{type(labels)} shape={labels_shape}",
            type(outputs),
        )
        if hasattr(outputs, "keys"):
            logging.error("Outputs keys: %s", list(outputs.keys()))
        elif isinstance(outputs, tuple):
            logging.error("Outputs tuple length: %d", len(outputs))
        raise ValueError("HuggingFace model did not return a tensor loss.")

    logits = logits.to(labels.device) if labels.device != logits.device else logits
    labels = labels.to(logits.device)

    vocab_size = logits.size(-1)
    if logits.ndim > 2:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if shift_logits.numel() > 0:
            logits = shift_logits
            labels = shift_labels

    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    valid_mask = labels_flat != -100
    if not torch.any(valid_mask):
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
    )
    return loss


class HuggingFaceTrainingStepStrategy(TrainingStepStrategy):
    """Performs forward/backward steps with optional gradient accumulation."""

    def __init__(self, gradient_accumulation_steps: Optional[int] = None):
        self.gradient_accumulation_steps = (
            int(gradient_accumulation_steps) if gradient_accumulation_steps else 1
        )

    def setup(self, context: TrainingContext) -> None:
        """Ensure gradient accumulation state is initialized."""
        context.state.setdefault("grad_accum_counter", 0)
        context.state.setdefault("grad_accum_loss_total", 0.0)
        context.state.setdefault("grad_accum_loss_count", 0)

    def get_accumulation_steps(self, context: TrainingContext) -> int:
        """Return configured gradient accumulation steps."""
        steps = context.config.get("gradient_accumulation_steps")
        if steps is None:
            steps = self.gradient_accumulation_steps
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            steps = 1
        return max(steps, 1)

    def training_step(
        self,
        model,
        optimizer,
        examples,
        labels,
        loss_criterion,  # pylint: disable=unused-argument
        context: TrainingContext,
    ):
        accum_steps = self.get_accumulation_steps(context)
        counter = int(context.state.get("grad_accum_counter", 0))

        if counter == 0:
            optimizer.zero_grad()

        batch_inputs = dict(examples)
        if labels is not None:
            batch_inputs["labels"] = labels
        batch_inputs.setdefault("return_dict", True)

        outputs = model(**batch_inputs)
        labels_tensor = batch_inputs.get("labels")
        loss = _resolve_hf_loss(outputs, labels_tensor)

        loss_for_backward = loss.div(accum_steps) if accum_steps > 1 else loss
        loss_for_backward.backward()

        counter += 1
        context.state["grad_accum_counter"] = counter
        loss_detached = loss.detach()
        context.state["grad_accum_loss_total"] = (
            context.state.get("grad_accum_loss_total", 0.0) + loss_detached.item()
        )
        context.state["grad_accum_loss_count"] = (
            context.state.get("grad_accum_loss_count", 0) + 1
        )

        should_step = counter >= accum_steps
        if not should_step and context.state.get("is_last_batch", False):
            should_step = counter > 0

        trainer = context.state.get("hf_trainer")

        if should_step:
            if trainer is not None:
                trainer._hf_on_pre_optimizer_step()

            optimizer.step()

            if trainer is not None:
                trainer._hf_on_optimizer_step()

            optimizer.zero_grad()

            loss_total = context.state.get("grad_accum_loss_total", 0.0)
            loss_count = max(context.state.get("grad_accum_loss_count", 0), 1)
            context.state["hf_loss_for_step"] = loss_total / loss_count
            context.state["grad_accum_loss_total"] = 0.0
            context.state["grad_accum_loss_count"] = 0
            context.state["optimizer_step_completed"] = True
            context.state["hf_optimizer_step_index"] = (
                context.state.get("hf_optimizer_step_index", 0) + 1
            )
            context.state["grad_accum_counter"] = 0
        else:
            context.state["optimizer_step_completed"] = False

        return loss_detached

    def finalize(self, model, optimizer, context: TrainingContext):
        """
        Flush any remaining accumulated gradients (e.g., partial final micro-batch).

        Returns a detached loss tensor representative of the accumulated step
        when a step is executed, otherwise None.
        """
        counter = int(context.state.get("grad_accum_counter", 0))
        if counter == 0:
            return None

        trainer = context.state.get("hf_trainer")
        if trainer is not None:
            trainer._hf_on_pre_optimizer_step()

        optimizer.step()

        if trainer is not None:
            trainer._hf_on_optimizer_step()

        optimizer.zero_grad()

        loss_total = context.state.get("grad_accum_loss_total", 0.0)
        loss_count = max(context.state.get("grad_accum_loss_count", 0), 1)
        average_loss = loss_total / loss_count if loss_count else 0.0
        context.state["hf_loss_for_step"] = average_loss
        context.state["grad_accum_loss_total"] = 0.0
        context.state["grad_accum_loss_count"] = 0
        context.state["grad_accum_counter"] = 0
        context.state["optimizer_step_completed"] = True
        context.state["hf_optimizer_step_index"] = (
            context.state.get("hf_optimizer_step_index", 0) + 1
        )

        if context.device is not None:
            loss_tensor = torch.tensor(average_loss, device=context.device)
        else:
            loss_tensor = torch.tensor(average_loss)
        return loss_tensor.detach()


class HuggingFaceTestingStrategy(TestingStrategy):
    """Evaluates HuggingFace models and reports perplexity based on loss."""

    def __init__(self, collate_fn: HuggingFaceCollateWrapper):
        self.collate_fn = collate_fn

    def test_model(self, model, config, testset, sampler, context: TrainingContext):
        batch_size = config.get("batch_size", 1)

        # Resolve sampler into a torch sampler when provided.
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                sampler_obj = sampler.get()
            else:
                sampler_obj = sampler
        else:
            sampler_obj = None

        data_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_obj,
            collate_fn=self.collate_fn,
        )

        model.to(context.device)
        model.eval()
        context.state["eval_loader"] = data_loader

        total_loss = 0.0
        total_weight = 0

        with torch.no_grad():
            for batch_inputs, labels in data_loader:
                batch_inputs = batch_inputs.to(context.device)
                if labels is not None:
                    labels = labels.to(context.device)
                    batch_inputs["labels"] = labels

                batch_inputs.setdefault("return_dict", True)
                outputs = model(**batch_inputs)
                loss = _resolve_hf_loss(outputs, labels)

                if labels is not None:
                    weight = labels.ne(-100).sum().item()
                    if weight == 0:
                        continue
                else:
                    weight = 1

                total_loss += loss.item() * weight
                total_weight += weight

        model.train()
        context.state.pop("eval_loader", None)

        if total_weight == 0:
            return float("inf")

        avg_loss = total_loss / total_weight
        try:
            return math.exp(avg_loss)
        except OverflowError:
            return float("inf")


def _split_callback_types(
    callbacks: Optional[Sequence[Union[type, object]]],
) -> Tuple[Sequence[Union[type, object]], Sequence[Union[type, object]]]:
    """Separate HuggingFace callbacks from Plato trainer callbacks."""
    if not callbacks:
        return [], []

    hf_callbacks = []
    plato_callbacks = []
    for callback in callbacks:
        callback_cls = callback if isinstance(callback, type) else callback.__class__
        if isinstance(callback_cls, type) and issubclass(
            callback_cls, HFTrainerCallback
        ):
            hf_callbacks.append(callback)
        elif isinstance(callback, HFTrainerCallback):
            hf_callbacks.append(callback)
        else:
            plato_callbacks.append(callback)
    return hf_callbacks, plato_callbacks


class HuggingFaceCallbackBridge(PlatoTrainerCallback):
    """Adapter that invokes HuggingFace callbacks via Plato callback events."""

    def __init__(self, trainer: "Trainer"):
        self._trainer = trainer

    def on_train_run_start(self, trainer, config, **kwargs):
        self._trainer._hf_on_train_begin(config)

    def on_train_run_end(self, trainer, config, **kwargs):
        self._trainer._hf_on_train_end()

    def on_train_epoch_start(self, trainer, config, **kwargs):
        self._trainer._hf_on_epoch_begin()

    def on_train_epoch_end(self, trainer, config, **kwargs):
        self._trainer._hf_on_epoch_end()

    def on_train_step_start(self, trainer, config, batch, **kwargs):
        counter = trainer.context.state.get("grad_accum_counter", 0)
        if counter == 0:
            self._trainer._hf_on_step_begin(batch)

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        if not trainer.context.state.get("optimizer_step_completed", True):
            return
        self._trainer._hf_on_step_end(batch, loss)
        self._trainer._hf_handle_control_flags()


class Trainer(ComposableTrainer):
    """Composable HuggingFace trainer built on Plato's strategy API."""

    def __init__(self, model=None, callbacks=None):
        hf_callbacks, plato_callbacks = _split_callback_types(callbacks)

        self._hf_callbacks: list[TrainerCallback] = []
        self._hf_bridge: Optional[HuggingFaceCallbackBridge] = None
        self._hf_state = TrainerState()
        self._hf_control = TrainerControl()
        self._hf_steps_per_epoch: Optional[int] = None

        parser = HfArgumentParser(TrainingArguments)
        (self.training_args,) = parser.parse_args_into_dataclasses(
            args=[
                "--output_dir=" + Config.params["checkpoint_path"],
                "--report_to=none",
            ]
        )

        model_name = Config().trainer.model_name
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }
        self.config = AutoConfig.from_pretrained(model_name, **config_kwargs)

        tokenizer_kwargs = {
            "cache_dir": None,
            "use_fast": True,
            "revision": "main",
            "use_auth_token": None,
        }
        if "llama" in model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name, config=self.config, **tokenizer_kwargs
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, config=self.config, **tokenizer_kwargs
            )

        grad_accum_steps = getattr(Config().trainer, "gradient_accumulation_steps", 1)
        try:
            grad_accum_steps = int(grad_accum_steps)
        except (TypeError, ValueError):
            grad_accum_steps = 1
        self._gradient_accumulation_steps = max(grad_accum_steps, 1)
        self._collate_wrapper = HuggingFaceCollateWrapper(self.tokenizer)
        self.training_args.gradient_accumulation_steps = (
            self._gradient_accumulation_steps
        )

        super().__init__(
            model=model,
            callbacks=plato_callbacks,
            loss_strategy=None,
            optimizer_strategy=None,
            training_step_strategy=HuggingFaceTrainingStepStrategy(
                gradient_accumulation_steps=self._gradient_accumulation_steps
            ),
            lr_scheduler_strategy=None,
            model_update_strategy=None,
            data_loader_strategy=CustomCollateFnDataLoaderStrategy(
                collate_fn=self._collate_wrapper,
                num_workers=0,
                pin_memory=True,
            ),
            testing_strategy=HuggingFaceTestingStrategy(self._collate_wrapper),
        )

        if hf_callbacks:
            self.add_callbacks(hf_callbacks)

        if hasattr(self.model, "loss_type"):
            self.model.loss_type = "ForCausalLM"

        # Ensure model checkpoints can be saved when model names include slashes.
        params = Config().params
        try:
            model_path = params["model_path"]
        except (TypeError, KeyError):
            model_path = None
        sub_dir = os.path.dirname(model_name)
        if model_path and sub_dir:
            os.makedirs(os.path.join(model_path, sub_dir), exist_ok=True)
        self.context.state["hf_trainer"] = self
        self.context.state["grad_accum_counter"] = 0
        self.context.state["grad_accum_loss_total"] = 0.0
        self.context.state["grad_accum_loss_count"] = 0
        self._hf_pending_keys = (
            "save",
            "evaluate",
            "log",
            "stop_epoch",
            "stop_training",
        )
        self._hf_pending_actions = {key: False for key in self._hf_pending_keys}
        self._hf_pending_log_data = None

    def add_callbacks(self, callbacks: Sequence[Union[type, object]]):
        """
        Add callbacks to the HuggingFace trainer.

        HuggingFace TrainerCallbacks are stored for future integration, while
        Plato trainer callbacks are registered with the composable callback
        handler.
        """
        hf_callbacks, plato_callbacks = _split_callback_types(callbacks)

        for callback in hf_callbacks:
            instance = callback() if isinstance(callback, type) else callback
            callback_cls = instance.__class__
            if not isinstance(instance, HFTrainerCallback):
                raise ValueError(
                    f"HuggingFace trainer expects subclass of {HFTrainerCallback}, got {callback_cls}."
                )
            self._hf_callbacks.append(instance)

        if self._hf_callbacks:
            self._ensure_hf_bridge()

        if plato_callbacks:
            self.callback_handler.add_callbacks(plato_callbacks)

    def _ensure_hf_bridge(self):
        if self._hf_bridge is None:
            self._hf_bridge = HuggingFaceCallbackBridge(self)
            self.callback_handler.add_callbacks([self._hf_bridge])

    def train_model(self, config, trainset, sampler, **kwargs):
        """Update HuggingFace training arguments before delegating to strategies."""
        self.training_args.num_train_epochs = config["epochs"]
        self.training_args.per_device_train_batch_size = config["batch_size"]
        accum_steps = config.get(
            "gradient_accumulation_steps", self._gradient_accumulation_steps
        )
        try:
            accum_steps = int(accum_steps)
        except (TypeError, ValueError):
            accum_steps = 1
        accum_steps = max(accum_steps, 1)
        self.training_args.gradient_accumulation_steps = accum_steps
        self._gradient_accumulation_steps = accum_steps
        if hasattr(self.training_step_strategy, "gradient_accumulation_steps"):
            self.training_step_strategy.gradient_accumulation_steps = accum_steps
        self.context.state["grad_accum_counter"] = 0
        self.context.state["grad_accum_loss_total"] = 0.0
        self.context.state["grad_accum_loss_count"] = 0
        self.context.state["hf_optimizer_step_index"] = 0
        if self._hf_callbacks:
            self._hf_state = TrainerState()
            self._hf_control = TrainerControl()
            self._hf_state.num_train_epochs = config.get("epochs", 1)
            self._hf_state.max_steps = 0
            self._hf_steps_per_epoch = None
            self._hf_pending_actions = {key: False for key in self._hf_pending_keys}
            self._hf_pending_log_data = None
        return super().train_model(config, trainset, sampler, **kwargs)

    def test_model(self, config, testset, sampler=None, **kwargs):
        """Update HuggingFace evaluation batch size before testing."""
        self.training_args.per_device_eval_batch_size = config.get("batch_size", 1)
        result = super().test_model(config, testset, sampler=sampler, **kwargs)
        if self._hf_callbacks:
            metrics = (
                {"perplexity": result} if isinstance(result, (int, float)) else result
            )
            self._hf_on_evaluate(metrics)
        return result

    def save_model(self, filename=None, location=None):
        """Save checkpoint and inform HuggingFace callbacks."""
        super().save_model(filename=filename, location=location)
        if self._hf_callbacks:
            self._hf_call_callbacks("on_save", model=self.model)
            self._hf_handle_control_flags()

    # --- HuggingFace callback integration helpers ---

    def _hf_call_callbacks(self, method: str, **kwargs):
        for callback in self._hf_callbacks:
            handler = getattr(callback, method, None)
            if handler is None:
                continue
            result = handler(
                self.training_args, self._hf_state, self._hf_control, **kwargs
            )
            if isinstance(result, TrainerControl):
                self._hf_control = result

    def _hf_on_train_begin(self, config):
        self._hf_control._new_training()
        self._hf_state.global_step = 0
        self._hf_state.epoch = 0
        self._hf_pending_actions = {key: False for key in self._hf_pending_keys}
        self._hf_pending_log_data = None
        self._hf_call_callbacks(
            "on_train_begin",
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataloader=self.train_loader,
        )

    def _hf_on_train_end(self):
        self._hf_call_callbacks(
            "on_train_end",
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataloader=self.train_loader,
        )
        self._hf_handle_control_flags()

    def _hf_update_training_metadata(self):
        if self.train_loader is None:
            return
        if self._hf_steps_per_epoch is None:
            try:
                steps = len(self.train_loader)
            except TypeError:
                steps = None
            if steps:
                accum_steps = self.training_step_strategy.get_accumulation_steps(
                    self.context
                )
                if accum_steps > 0:
                    steps = math.ceil(steps / accum_steps)
                self._hf_steps_per_epoch = steps
                self._hf_state.max_steps = steps * self._hf_state.num_train_epochs
                batch_size = getattr(self.train_loader, "batch_size", None)
                self._hf_state.train_batch_size = batch_size

    def _hf_on_epoch_begin(self):
        self._hf_control._new_epoch()
        self._hf_update_training_metadata()
        current_epoch = max(self.current_epoch - 1, 0)
        self._hf_state.epoch = float(current_epoch)
        self._hf_call_callbacks(
            "on_epoch_begin",
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_loader,
        )
        self._hf_handle_control_flags()

    def _hf_on_epoch_end(self):
        self._hf_state.epoch = float(self.current_epoch)
        self._hf_call_callbacks(
            "on_epoch_end",
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_loader,
        )
        self._hf_handle_control_flags()

    def _hf_on_step_begin(self, batch_index: int):
        self._hf_control._new_step()
        self._hf_call_callbacks(
            "on_step_begin",
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_loader,
            step=batch_index,
        )
        self._hf_handle_control_flags()

    def _hf_on_pre_optimizer_step(self):
        self._hf_call_callbacks(
            "on_pre_optimizer_step",
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

    def _hf_on_optimizer_step(self):
        self._hf_call_callbacks(
            "on_optimizer_step",
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

    def _hf_on_step_end(self, batch_index: int, loss: torch.Tensor):
        if self._hf_steps_per_epoch:
            step_index = self.context.state.get("hf_optimizer_step_index")
            if step_index is None:
                progress = (batch_index + 1) / self._hf_steps_per_epoch
            else:
                progress = step_index / self._hf_steps_per_epoch
            progress = min(progress, 1.0)
            self._hf_state.epoch = float(self.current_epoch - 1 + progress)
        self._hf_state.global_step += 1
        loss_override = self.context.state.pop("hf_loss_for_step", None)
        if loss_override is not None:
            loss_value = float(loss_override)
        else:
            loss_value = loss.item() if hasattr(loss, "item") else float(loss)
        log_entry = {
            "loss": float(loss_value),
            "step": self._hf_state.global_step,
            "epoch": self._hf_state.epoch,
        }
        current_lr = self._get_current_lr()
        if current_lr is not None:
            log_entry["learning_rate"] = current_lr
        self._hf_state.log_history.append(log_entry)
        self._hf_pending_log_data = log_entry
        self._hf_call_callbacks(
            "on_step_end",
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_loader,
            metrics={"loss": loss_value},
        )

    def _hf_on_evaluate(self, metrics):
        self._hf_call_callbacks(
            "on_evaluate",
            model=self.model,
            tokenizer=self.tokenizer,
            metrics=metrics,
            eval_dataloader=self.context.state.get("eval_loader"),
        )
        self._hf_handle_control_flags()

    def _hf_handle_control_flags(self):
        if not self._hf_callbacks:
            return

        control = self._hf_control

        if control.should_save:
            self._hf_pending_actions["save"] = True
            control.should_save = False

        if control.should_evaluate:
            self._hf_pending_actions["evaluate"] = True
            control.should_evaluate = False

        if control.should_log:
            self._hf_pending_actions["log"] = True
            control.should_log = False

        if control.should_epoch_stop:
            self._hf_pending_actions["stop_epoch"] = True
            control.should_epoch_stop = False

        if control.should_training_stop:
            self._hf_pending_actions["stop_training"] = True
            self._hf_pending_actions["stop_epoch"] = True
            control.should_training_stop = False

    def _consume_control_flags(self):
        if not self._hf_callbacks:
            return {}

        actions = {
            key: bool(self._hf_pending_actions.get(key))
            for key in self._hf_pending_keys
        }
        self._hf_pending_actions = {key: False for key in self._hf_pending_keys}
        return actions

    def _handle_control_evaluate(self):
        metrics = {}
        try:
            datasource = datasources_registry.get(client_id=self.client_id)
            testset = datasource.get_test_set()
            metrics_value = self.testing_strategy.test_model(
                self.model,
                self.context.config,
                testset,
                None,
                self.context,
            )
            metrics = {"perplexity": metrics_value}
        except Exception as exc:
            logging.warning(
                "HuggingFace trainer failed to run evaluation requested by callback: %s",
                exc,
            )
        finally:
            self._hf_on_evaluate(metrics)

    def _handle_control_log(self):
        logs = {}
        if self._hf_pending_log_data is not None:
            logs = dict(self._hf_pending_log_data)
        elif self._hf_state.log_history:
            logs = dict(self._hf_state.log_history[-1])
        else:
            last_loss = self.context.state.get("last_loss")
            if last_loss is not None:
                logs = {"loss": float(last_loss)}

        if logs and "learning_rate" not in logs:
            current_lr = self._get_current_lr()
            if current_lr is not None:
                logs["learning_rate"] = current_lr

        self._hf_call_callbacks(
            "on_log",
            model=self.model,
            tokenizer=self.tokenizer,
            logs=logs,
        )
        self._hf_pending_log_data = None

    def _get_current_lr(self):
        if self.optimizer is None:
            return None
        for group in getattr(self.optimizer, "param_groups", []):
            lr = group.get("lr")
            if lr is not None:
                return lr
        return None
