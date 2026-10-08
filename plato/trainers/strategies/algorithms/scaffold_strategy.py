"""SCAFFOLD Algorithm 1 / Option II with fixed positive optimizer LR.

For vanilla SGD: y <- y - eta*(g-ci+c), then
ci_new = ci-c+(x-y)/(K*eta). K counts completed optimizer updates.
The server uses c <- c + sum(delta_ci)/N, where N is the population.
Other optimizers receive an additive post-update control correction; this
extension does not claim the paper's gradient-average identity.

Reference: https://arxiv.org/pdf/1910.06378v4 (Algorithm 1, equations 3-5).
Historical control dictionaries remain structurally loadable, but corrected
continuation differs numerically. Start a new run with consistent server and
client controls to obtain a fresh reference run.
"""

import logging
import math
import os
import pickle
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from plato.config import Config
from plato.trainers.strategies.base import ModelUpdateStrategy, TrainingContext
from plato.utils.checkpoint_paths import checkpoint_name, checkpoint_path


def validate_control_variates(
    model: nn.Module, controls: Any
) -> OrderedDict[str, torch.Tensor]:
    """Copy finite trainable controls, projecting known historical state extras."""
    if not isinstance(controls, Mapping):
        raise ValueError("SCAFFOLD controls must be a parameter mapping.")
    parameters = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    unknown = set(controls) - set(model.state_dict())
    missing = set(parameters) - set(controls)
    if unknown or missing:
        raise ValueError(
            f"SCAFFOLD control keys mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}."
        )
    result = OrderedDict()
    for name, parameter in parameters.items():
        value = controls[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"SCAFFOLD control for {name} must be finite and match its parameter shape."
            )
        copied = value.detach().to(device="cpu", dtype=parameter.dtype).clone()
        if not torch.isfinite(copied).all():
            raise ValueError(
                f"SCAFFOLD control for {name} overflows the parameter dtype."
            )
        result[name] = copied
    return result


class SCAFFOLDUpdateStrategy(ModelUpdateStrategy):
    """Own one logical client's controls and fixed-rate Option II round state.

    The actual optimizer in context.state['optimizer'] owns rate and parameter
    selection. Included parameters with no gradient receive the zero-gradient
    control correction. Excluded parameters retain ci and emit zero delta.
    Missing server controls warn and initialize zero only for standalone use;
    the federated example requires a current, validated inbound control payload.
    """

    def __init__(self, save_path: str | None = None):
        self.save_path = save_path
        self.client_control_variate: OrderedDict[str, torch.Tensor] | None = None
        self.server_control_variate: OrderedDict[str, torch.Tensor] | None = None
        self.global_model_weights: OrderedDict[str, torch.Tensor] | None = None
        self.client_control_variate_path: str | None = None
        self.local_steps = 0
        self.learning_rate: float | None = None
        self._step_hook = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._participating: OrderedDict[str, nn.Parameter] = OrderedDict()
        self._parameter_names: dict[int, str] = {}
        self._executed_rate: float | None = None
        self._run_completed = False
        self._result_accepted = False
        self._has_pending_controls = False
        self._controls_before_acceptance: OrderedDict[str, torch.Tensor] | None = None

    @staticmethod
    def _model(context: TrainingContext) -> nn.Module:
        if context.model is None:
            raise ValueError("SCAFFOLD requires a model in the training context.")
        return context.model

    def setup(self, context: TrainingContext) -> None:
        root = (
            self.save_path
            if self.save_path is not None
            else Config().params["model_path"]
        )
        os.makedirs(root, exist_ok=True)
        self.client_control_variate_path = checkpoint_path(
            root, checkpoint_name("scaffold_cv", context.client_id, suffix=".pkl")
        )
        # Construction-time client 0 must never become another client's state.
        if context.client_id != 0:
            self._load_client_controls(context, root)

    def _load_client_controls(self, context: TrainingContext, root: str) -> None:
        canonical = self.client_control_variate_path
        if canonical is None:
            raise RuntimeError("SCAFFOLD client state path is not initialized.")
        candidates = [canonical]
        # Exact historical example filename, contained under the configured root.
        legacy_name = (
            f"{Config().trainer.model_name}_{context.client_id}_control_variate.pth"
        )
        try:
            candidates.append(checkpoint_path(root, legacy_name))
        except ValueError:
            pass
        # Bounded read-only migration of the old prefix path for this nonzero ID.
        candidates.append(f"{root}scaffold_cv_{context.client_id}.pkl")
        for path in candidates:
            if os.path.isfile(path):
                with open(path, "rb") as state_file:
                    controls = pickle.load(state_file)
                self.client_control_variate = validate_control_variates(
                    self._model(context), controls
                )
                return

    def on_client_id_changed(self, context: TrainingContext) -> None:
        self.on_train_cleanup(context, successful=False)
        self.client_control_variate = None
        self.server_control_variate = None
        self.global_model_weights = None
        context.state.pop("server_control_variate", None)
        self.setup(context)

    def on_train_start(self, context: TrainingContext) -> None:
        self.on_train_cleanup(context, successful=False)
        self.local_steps = 0
        self.learning_rate = None
        self._executed_rate = None
        self._run_completed = False
        model = self._model(context)
        self._parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        if self.client_control_variate is None:
            self.client_control_variate = OrderedDict(
                (name, torch.zeros_like(parameter, device="cpu"))
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        self.client_control_variate = validate_control_variates(
            model, self.client_control_variate
        )
        controls = context.state.get("server_control_variate")
        if controls is None:
            logging.warning(
                "SCAFFOLD standalone run has no server controls; using zeros."
            )
            controls = OrderedDict(
                (name, torch.zeros_like(value))
                for name, value in self.client_control_variate.items()
            )
        self.server_control_variate = validate_control_variates(model, controls)
        self.global_model_weights = OrderedDict(
            (name, parameter.detach().cpu().clone())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )

    def before_step(self, context: TrainingContext) -> None:
        """Install an actual-step hook once; final accumulation uses the same hook."""
        optimizer = context.state.get("optimizer")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(
                "SCAFFOLD requires context.state['optimizer'] with the actual optimizer."
            )
        if self._optimizer is optimizer:
            return
        if self._optimizer is not None:
            raise ValueError("SCAFFOLD optimizer cannot change within a local round.")
        self._optimizer = optimizer
        self._step_hook = optimizer.register_step_pre_hook(self._capture_update)
        vanilla = isinstance(optimizer, torch.optim.SGD) and all(
            not group.get("momentum", 0)
            and not group.get("weight_decay", 0)
            and not group.get("maximize", False)
            for group in optimizer.param_groups
        )
        if not vanilla:
            logging.warning(
                "SCAFFOLD uses an additive-control optimizer extension for this run; "
                "only vanilla fixed-rate SGD has the paper gradient-average identity."
            )

    def _capture_update(self, optimizer, args, kwargs) -> None:
        participating = OrderedDict()
        rates = []
        for group in optimizer.param_groups:
            owned = []
            for parameter in group["params"]:
                if parameter.requires_grad:
                    name = self._parameter_names.get(id(parameter))
                    if name is None:
                        raise ValueError(
                            "SCAFFOLD optimizer contains a parameter outside the model."
                        )
                    if name in participating:
                        raise ValueError(
                            "SCAFFOLD optimizer repeats a model parameter."
                        )
                    participating[name] = parameter
                    owned.append(parameter)
            if owned:
                rate = group["lr"]
                if isinstance(rate, (torch.Tensor, bool)) or not isinstance(
                    rate, (int, float)
                ):
                    raise ValueError(
                        "SCAFFOLD requires a positive finite scalar optimizer LR."
                    )
                if not math.isfinite(rate) or rate <= 0:
                    raise ValueError(
                        "SCAFFOLD requires a positive finite scalar optimizer LR."
                    )
                rates.append(float(rate))
        if not rates or any(rate != rates[0] for rate in rates):
            raise ValueError(
                "SCAFFOLD requires equal positive LR across participating optimizer groups."
            )
        rate = rates[0]
        if self.learning_rate is not None and self.learning_rate != rate:
            raise ValueError(
                "SCAFFOLD optimizer LR must remain constant within a local round."
            )
        if self.local_steps and set(participating) != set(self._participating):
            raise ValueError(
                "SCAFFOLD optimizer parameter ownership changed within the round."
            )
        self.learning_rate = self._executed_rate = rate
        self._participating = participating

    def after_step(self, context: TrainingContext) -> None:
        """Apply -eta*(c-ci) once after a completed optimizer update."""
        if context.state.get("optimizer_step_completed", True) is False:
            self._executed_rate = None
            return
        rate = self._executed_rate
        if (
            rate is None
            or self.server_control_variate is None
            or self.client_control_variate is None
        ):
            raise RuntimeError(
                "SCAFFOLD correction requires a captured executed optimizer step."
            )
        with torch.no_grad():
            for name, parameter in self._participating.items():
                correction = (
                    self.server_control_variate[name]
                    - self.client_control_variate[name]
                )
                parameter.add_(correction.to(parameter), alpha=-rate)
        self.local_steps += 1
        self._executed_rate = None

    def on_train_end(self, context: TrainingContext) -> None:
        """Stage ci_old-c+(x-y)/(K*eta) until the whole result is accepted."""
        model = self._model(context)
        old = self.client_control_variate
        server = self.server_control_variate
        initial = self.global_model_weights
        if old is None or server is None or initial is None:
            raise RuntimeError("SCAFFOLD round state is not initialized.")
        new = OrderedDict((name, value.clone()) for name, value in old.items())
        if self.local_steps:
            if self.learning_rate is None:
                raise RuntimeError("SCAFFOLD completed updates have no executed LR.")
            for name, parameter in model.named_parameters():
                if name in self._participating:
                    new[name] = (
                        old[name]
                        - server[name]
                        + (initial[name] - parameter.detach().cpu())
                        / (self.local_steps * self.learning_rate)
                    )
        new = validate_control_variates(model, new)
        delta = OrderedDict((name, new[name] - value) for name, value in old.items())
        self._controls_before_acceptance = old
        self._has_pending_controls = True
        self.client_control_variate = new
        context.state["client_control_variate_delta"] = delta
        self._run_completed = True
        self._result_accepted = False

    def on_train_result_accepted(self, context: TrainingContext) -> None:
        """Atomically persist controls only after model/result acceptance.

        Workers export provisional controls without touching the canonical
        file. Direct training commits after callbacks; the parent commits a
        spawned result after validating its current token and loading both parts.
        """
        if not self._run_completed or not self._has_pending_controls:
            raise RuntimeError("SCAFFOLD has no current result to accept.")
        path = self.client_control_variate_path
        if path is None:
            raise RuntimeError("SCAFFOLD client state path is not initialized.")
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".scaffold_cv_", dir=os.path.dirname(path)
            )
            with os.fdopen(descriptor, "wb") as state_file:
                pickle.dump(self.client_control_variate, state_file)
            os.replace(temporary, path)
        except BaseException:
            self.on_train_cleanup(context, successful=False)
            raise
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.remove(temporary)
        self._has_pending_controls = False
        self._controls_before_acceptance = None
        self._result_accepted = True

    def on_train_cleanup(self, context: TrainingContext, successful: bool) -> None:
        if self._step_hook is not None:
            self._step_hook.remove()
        self._step_hook = None
        self._optimizer = None
        if not successful:
            if self._has_pending_controls:
                self.client_control_variate = self._controls_before_acceptance
            self._has_pending_controls = False
            self._controls_before_acceptance = None
            context.state.pop("client_control_variate_delta", None)
            self._run_completed = False
            self._result_accepted = False
        self._executed_rate = None
        self._participating = OrderedDict()
        self._parameter_names = {}

    def get_update_payload(self, context: TrainingContext) -> dict[str, Any]:
        delta = context.state.get("client_control_variate_delta")
        if not self._run_completed or not self._result_accepted or delta is None:
            raise RuntimeError(
                "SCAFFOLD has no successful current-round control delta."
            )
        return {"control_variate_delta": delta}

    @property
    def requires_worker_state(self) -> bool:
        return True

    def get_worker_state(self, context: TrainingContext) -> dict[str, Any]:
        delta = context.state.get("client_control_variate_delta")
        if not self._run_completed or delta is None:
            raise RuntimeError("SCAFFOLD worker has no completed current result.")
        return {
            "client_control_variate": self.client_control_variate,
            "delta": delta,
            "local_steps": self.local_steps,
            "learning_rate": self.learning_rate,
        }

    def load_worker_state(self, state: Any, context: TrainingContext) -> None:
        if not isinstance(state, dict):
            raise ValueError("SCAFFOLD worker control state is missing.")
        controls = validate_control_variates(
            self._model(context), state.get("client_control_variate")
        )
        delta = validate_control_variates(self._model(context), state.get("delta"))
        steps, rate = state.get("local_steps"), state.get("learning_rate")
        if (
            not isinstance(steps, int)
            or steps < 0
            or (
                steps
                and (
                    not isinstance(rate, (int, float))
                    or not math.isfinite(rate)
                    or rate <= 0
                )
            )
        ):
            raise ValueError("SCAFFOLD worker update counters are invalid.")
        self._controls_before_acceptance = self.client_control_variate
        self._has_pending_controls = True
        self.client_control_variate = controls
        context.state["client_control_variate_delta"] = delta
        self.local_steps, self.learning_rate = steps, rate
        self._run_completed = True
        self._result_accepted = False

    def teardown(self, context: TrainingContext) -> None:
        self.on_train_cleanup(context, successful=False)
        self.server_control_variate = self.global_model_weights = None


class SCAFFOLDUpdateStrategyV2(SCAFFOLDUpdateStrategy):
    """Compatibility name for corrected Option II, not paper Option I.

    The historical raw-displacement accumulator did not evaluate the extra
    gradient pass required by Option I. This subclass shares Option II's
    executed-step, fixed-rate and persistence contracts.
    """
