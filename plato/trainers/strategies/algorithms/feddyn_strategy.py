"""FedDyn's corrected objective and server-dispatched provisional histories.

J=F+alpha_i<h,w>+alpha_i/2||w-x||²; h is cumulative displacement, not a
measured gradient. Reference: arXiv:2111.04263v1, Algorithm 1, and author
commit 48a19fac440ef079ce563da8e0c2896f8256fef9. Old files are inspection only.
"""

from __future__ import annotations

import copy
import math
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch

from plato.config import Config
from plato.trainers.strategies.base import (
    LossCriterionStrategy,
    ModelUpdateStrategy,
    TrainingContext,
)
from plato.utils.checkpoint_paths import checkpoint_name, checkpoint_path


def finite_real(value: Any, name: str, *, positive: bool = False) -> float:
    """Validate finite nonnegative real coefficients, excluding bool."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"FedDyn {name} must be a finite real number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"FedDyn {name} must be finite.") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            f"FedDyn {name} must be finite and nonnegative"
            + (" and nonzero." if positive else ".")
        )
    return result


def positive_integer(value: Any, name: str) -> int:
    """Validate an authoritative positive integer count or identity."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"FedDyn {name} must be a positive integer.")
    return int(value)


def population_counts(counts: Any, population: int) -> list[int]:
    """Validate the full fixed population vector, never numeric labels."""
    if not isinstance(counts, (list, tuple)) or len(counts) != population:
        raise ValueError("FedDyn sample counts must cover the full population.")
    result = [positive_integer(n, "sample count") for n in counts]
    try:
        total = float(sum(result))
    except OverflowError as exc:
        raise ValueError("FedDyn sample count sum overflows.") from exc
    if not math.isfinite(total):
        raise ValueError("FedDyn sample count sum must be finite.")
    return result


def effective_alpha(alpha: float, population: int, counts: Any, client_id: int):
    """Return alpha/q_i from full-population realized counts."""
    population = positive_integer(population, "population")
    client_id = positive_integer(client_id, "client ID")
    if client_id > population:
        raise ValueError("FedDyn client ID is outside the population.")
    counts = population_counts(counts, population)
    return finite_real(
        alpha * (sum(counts) / (population * counts[client_id - 1])), "effective alpha"
    )


def settings_from_config() -> dict[str, Any]:
    """Validate the example's fixed synchronous plain-SGD configuration."""
    c = Config()
    n = positive_integer(c.clients.total_clients, "population")
    per_round = positive_integer(c.clients.per_round, "per_round")
    if per_round > n:
        raise ValueError("FedDyn per_round exceeds the population.")
    alpha = finite_real(
        getattr(c.algorithm, "alpha_coef", getattr(c.algorithm, "feddyn_alpha", 0.01)),
        "alpha",
        positive=True,
    )
    mode = getattr(c.algorithm, "feddyn_weighting", "uniform")
    if mode not in ("uniform", "sample"):
        raise ValueError("FedDyn weighting must be uniform or sample.")
    raw_counts = getattr(c.algorithm, "feddyn_sample_counts", None)
    if mode == "uniform" and hasattr(c.algorithm, "feddyn_sample_counts"):
        raise ValueError("FedDyn uniform mode does not accept sample counts.")
    counts = population_counts(raw_counts, n) if mode == "sample" else None
    if counts is not None:
        for i in range(1, n + 1):
            finite_real(
                effective_alpha(alpha, n, counts, i), "effective alpha", positive=True
            )
    if getattr(c.algorithm, "type", None) != "fedavg":
        raise ValueError("FedDyn requires the underlying fedavg model exchange.")
    if (
        getattr(c.server, "synchronous", True) is not True
        or any(
            getattr(c.server, k, False)
            for k in ("request_update", "asynchronous", "fedbuff")
        )
        or hasattr(c.algorithm, "total_silos")
    ):
        raise ValueError("FedDyn supports synchronous non-cross-silo rounds only.")
    if getattr(c.trainer, "type", "basic") != "basic":
        raise ValueError("FedDyn requires the ordinary torch trainer, not DP.")
    if getattr(c.trainer, "optimizer", None) != "SGD":
        raise ValueError("FedDyn supports plain SGD only.")
    if any(
        getattr(c.trainer, k, False)
        for k in (
            "amp",
            "use_amp",
            "mixed_precision",
            "gradient_clip",
            "max_grad_norm",
            "gradient_clip_val",
            "gradient_clipping",
            "differential_privacy",
        )
    ):
        raise ValueError("FedDyn does not support AMP, clipping or DP.")
    if hasattr(c.trainer, "lr_scheduler"):
        raise ValueError("FedDyn requires fixed learning rate without a scheduler.")
    positive_integer(c.trainer.epochs, "epochs")
    positive_integer(c.trainer.batch_size, "batch size")
    accumulation = positive_integer(
        getattr(c.trainer, "gradient_accumulation_steps", 1), "accumulation steps"
    )
    params = c.parameters.optimizer._asdict()
    lr = finite_real(params.get("lr"), "learning rate", positive=True)
    for key in ("momentum", "dampening", "weight_decay"):
        if finite_real(params.get(key, 0), key) != 0:
            raise ValueError(f"FedDyn plain SGD requires {key}=0.")
    if params.get("nesterov", False) or params.get("maximize", False):
        raise ValueError("FedDyn plain SGD forbids nesterov/maximize.")
    return dict(
        population=n,
        per_round=per_round,
        base_alpha=alpha,
        weighting=mode,
        sample_counts=counts,
        lr=lr,
        accumulation_steps=accumulation,
    )


def model_schema(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    """Fingerprint fixed trainability and exact full transport shape/dtype."""
    if not isinstance(model, torch.nn.Module):
        raise ValueError("FedDyn requires an ordinary torch model.")
    parameters = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in parameters}) != len(parameters):
        raise ValueError("FedDyn does not support shared parameter aliases.")
    q = {name for name, p in parameters if p.requires_grad}
    if not q:
        raise ValueError("FedDyn needs a nonempty fixed trainable parameter set.")
    result = {}
    for name, v in model.state_dict().items():
        if (
            not isinstance(v, torch.Tensor)
            or v.layout != torch.strided
            or v.is_complex()
            or not torch.isfinite(v).all()
            or (v.is_floating_point() and v.dtype not in (torch.float32, torch.float64))
        ):
            raise ValueError(
                f"FedDyn model tensor {name} must be dense/finite float32/64 or static integral."
            )
        result[name] = dict(
            shape=list(v.shape), dtype=str(v.dtype), trainable=name in q
        )
    return result


def trainable_reference(model: torch.nn.Module):
    model_schema(model)
    return OrderedDict(
        (n, p.detach()) for n, p in model.named_parameters() if p.requires_grad
    )


def context_model(context: TrainingContext) -> torch.nn.Module:
    if context.model is None:
        raise ValueError("FedDyn context must contain an ordinary torch model.")
    return context.model


def validate_tensors(values: Any, reference: Mapping, name: str):
    """Validate exact keys/shapes/dtypes and take independent CPU ownership."""
    if not isinstance(values, Mapping) or set(values) != set(reference):
        raise ValueError(f"FedDyn {name} keys must match the exact model scope.")
    result = OrderedDict()
    for k, baseline in reference.items():
        v = values[k]
        if (
            not isinstance(v, torch.Tensor)
            or v.layout != torch.strided
            or v.shape != baseline.shape
            or v.dtype != baseline.dtype
            or not torch.isfinite(v).all()
        ):
            raise ValueError(
                f"FedDyn {name} tensor {k} needs finite exact shape/dtype."
            )
        result[k] = v.detach().cpu().clone()
    return result


def same_state(actual: Any, expected: Any) -> bool:
    """Compare owned state without bool/integer or broadcast equivalence."""
    if isinstance(expected, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor)
            and actual.dtype == expected.dtype
            and actual.shape == expected.shape
            and torch.equal(actual.cpu(), expected.cpu())
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and {(type(k), k) for k in actual} == {(type(k), k) for k in expected}
            and all(same_state(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, (list, tuple)):
        return (
            type(actual) is type(expected)
            and len(actual) == len(expected)
            and all(same_state(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def validate_endpoint(model, endpoint, baseline, schema):
    """Reject mutable buffers/frozen values instead of averaging them."""
    if model_schema(model) != schema:
        raise ValueError("FedDyn model trainability/schema changed.")
    baseline = validate_tensors(baseline, model.state_dict(), "baseline")
    endpoint = validate_tensors(endpoint, baseline, "endpoint")
    for name, spec in schema.items():
        if not spec["trainable"] and not torch.equal(endpoint[name], baseline[name]):
            raise ValueError(
                f"FedDyn mutable buffer/frozen parameter {name} is unsupported."
            )
    return endpoint


def validate_identity(value, name):
    if not isinstance(value, str):
        raise ValueError(f"FedDyn {name} must be a UUID hex string.")
    try:
        parsed = uuid.UUID(hex=value)
    except ValueError as exc:
        raise ValueError(f"FedDyn {name} must be a UUID hex string.") from exc
    if parsed.hex != value:
        raise ValueError(f"FedDyn {name} must be canonical UUID hex.")
    return value


def validate_dispatch(model, payload, settings, client_id, round_id):
    """Validate the entire downlink before changing model or strategy state."""
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise ValueError(
            "FedDyn needs [full model, versioned round state]; use the dedicated example."
        )
    full, raw_metadata = payload
    meta = raw_metadata
    keys = {
        "version",
        "run_id",
        "round",
        "client_id",
        "dispatch_token",
        "weighting",
        "base_alpha",
        "effective_alpha",
        "expected_count_or_null",
        "history",
    }
    if not isinstance(meta, dict) or set(meta) != keys:
        raise ValueError("FedDyn dispatch metadata is incomplete or unknown.")
    if type(meta["version"]) is not int or meta["version"] != 1:
        raise ValueError("FedDyn dispatch version must be 1.")
    client_id = positive_integer(client_id, "client ID")
    if (
        client_id > settings["population"]
        or positive_integer(meta["client_id"], "client ID") != client_id
        or positive_integer(meta["round"], "round") != round_id
    ):
        raise ValueError("FedDyn dispatch client/round differs from active assignment.")
    validate_identity(meta["run_id"], "run ID")
    validate_identity(meta["dispatch_token"], "dispatch token")
    alpha = (
        effective_alpha(
            settings["base_alpha"],
            settings["population"],
            settings["sample_counts"],
            client_id,
        )
        if settings["weighting"] == "sample"
        else settings["base_alpha"]
    )
    if (
        meta["weighting"] != settings["weighting"]
        or finite_real(meta["base_alpha"], "alpha", positive=True)
        != settings["base_alpha"]
        or finite_real(meta["effective_alpha"], "effective alpha", positive=True)
        != alpha
    ):
        raise ValueError("FedDyn dispatch configuration differs from this session.")
    count = meta["expected_count_or_null"]
    if count is not None:
        positive_integer(count, "expected sample count")
    if (
        settings["sample_counts"] is not None
        and count != settings["sample_counts"][client_id - 1]
    ):
        raise ValueError(
            "FedDyn dispatch count differs from the full population vector."
        )
    schema = model_schema(model)
    full = validate_tensors(full, model.state_dict(), "downlink model")
    h = validate_tensors(meta["history"], trainable_reference(model), "history")
    return dict(
        metadata=copy.deepcopy({**meta, "history": h}), baseline=full, schema=schema
    )


class FedDynLossStrategy(LossCriterionStrategy):
    """Shifted quadratic loss; adaptive alpha requires full population counts."""

    def __init__(
        self,
        alpha: float = 0.01,
        base_loss_fn: Callable | None = None,
        adaptive_alpha: bool = False,
    ):
        self.alpha = finite_real(alpha, "alpha")
        if type(adaptive_alpha) is not bool:
            raise ValueError("FedDyn adaptive_alpha must be bool.")
        self.adaptive_alpha = adaptive_alpha
        self.base_loss_fn = base_loss_fn or torch.nn.CrossEntropyLoss()
        self.global_model_weights = self.cumulative_grad_vector = None
        self._alpha_for_run = None

    def setup(self, context: TrainingContext) -> None:
        model = context_model(context)
        q = trainable_reference(model)
        full = context.state.get("feddyn_global_weights", model.state_dict())
        self.global_model_weights = validate_tensors(
            {k: full[k] for k in q}, q, "loss baseline"
        )
        h = context.state.get(
            "feddyn_cumulative_grad", {k: torch.zeros_like(v) for k, v in q.items()}
        )
        self.cumulative_grad_vector = validate_tensors(h, q, "loss history")
        self._alpha_for_run = None

    def on_train_start(self, context: TrainingContext) -> None:
        self.setup(context)
        self._alpha_for_run = self._get_alpha_coefficient(None, context)

    def _get_alpha_coefficient(self, labels, context: TrainingContext):
        if not self.adaptive_alpha:
            return self.alpha
        m = context.state.get("feddyn_count_metadata")
        if not isinstance(m, dict):
            raise ValueError(
                "FedDyn adaptive alpha needs authoritative count metadata."
            )
        return effective_alpha(
            self.alpha, m.get("population"), m.get("sample_counts"), m.get("client_id")
        )

    def compute_loss(self, outputs, labels, context: TrainingContext):
        task = self.base_loss_fn(outputs, labels)
        alpha = (
            self._alpha_for_run
            if self._alpha_for_run is not None
            else self._get_alpha_coefficient(labels, context)
        )
        if alpha == 0:
            return task
        if self.global_model_weights is None or self.cumulative_grad_vector is None:
            raise ValueError("FedDyn loss needs setup before regularization.")
        regularizer = None
        for name, p in context_model(context).named_parameters():
            if p.requires_grad:
                x = self.global_model_weights[name].to(p.device)
                h = self.cumulative_grad_vector[name].to(p.device)
                term = p.new_tensor(alpha) * (
                    (h * p).sum() + (p - x).square().sum() / 2
                )
                regularizer = term if regularizer is None else regularizer + term
        return task + regularizer

    def on_client_id_changed(self, context: TrainingContext) -> None:
        self.global_model_weights = self.cumulative_grad_vector = None
        self._alpha_for_run = None

    teardown = on_client_id_changed


class FedDynUpdateStrategy(ModelUpdateStrategy):
    """One staged result per dispatch; no child/client canonical history writes.

    save_path is solely a read-only migration location. Corrected training needs
    the dedicated example's validated server dispatch and parent transaction.
    """

    def __init__(self, save_path: str | None = None):
        self.save_path = save_path
        self.global_model_weights = self.cumulative_grad_vector = None
        self.grad_vector_path = self.dispatch = self.result = self.schema = None
        self.accepted = False
        self.completed_steps = 0
        self._lr = None

    def setup(self, context: TrainingContext) -> None:
        self.schema = model_schema(context.model)
        root = (
            self.save_path
            if self.save_path is not None
            else Config().params["model_path"]
        )
        self.grad_vector_path = checkpoint_path(
            root, checkpoint_name("feddyn_grad", context.client_id, suffix=".pth")
        )

    def install_dispatch(self, state, context: TrainingContext) -> None:
        self.dispatch = copy.deepcopy(state)
        self.result = None
        self.accepted = False
        self.global_model_weights = copy.deepcopy(state["baseline"])
        self.cumulative_grad_vector = copy.deepcopy(state["metadata"]["history"])
        context.state["feddyn_dispatch"] = copy.deepcopy(state)

    def on_client_id_changed(self, context: TrainingContext) -> None:
        self.dispatch = self.result = None
        self.accepted = False
        self.global_model_weights = self.cumulative_grad_vector = None
        for key in list(context.state):
            if key.startswith("feddyn_"):
                context.state.pop(key)
        self.setup(context)

    def on_train_start(self, context: TrainingContext) -> None:
        if self.save_path is not None:
            raise ValueError(
                "FedDyn save_path is read-only inspection; live history uses server checkpoint_path."
            )
        if self.dispatch is None:
            raise ValueError(
                "FedDyn training needs a versioned server dispatch; use the dedicated example or explicitly warm-start a new run."
            )
        s = self.dispatch
        m = s["metadata"]
        if m["client_id"] != context.client_id or m["round"] != context.current_round:
            raise ValueError("FedDyn training has stale client/round dispatch state.")
        settings = settings_from_config()
        validate_dispatch(
            context_model(context),
            [s["baseline"], m],
            settings,
            context.client_id,
            context.current_round,
        )
        model = context_model(context)
        endpoint = validate_endpoint(
            model, model.state_dict(), s["baseline"], s["schema"]
        )
        if any(not torch.equal(v, s["baseline"][k]) for k, v in endpoint.items()):
            raise ValueError("FedDyn training must start at received global baseline.")
        self.schema = s["schema"]
        self.global_model_weights = copy.deepcopy(s["baseline"])
        self.cumulative_grad_vector = copy.deepcopy(m["history"])
        self.result = None
        self.accepted = False
        self.completed_steps = 0
        self._lr = None
        context.state["feddyn_global_weights"] = self.global_model_weights
        context.state["feddyn_cumulative_grad"] = self.cumulative_grad_vector
        context.state["feddyn_count_metadata"] = dict(
            population=settings["population"],
            sample_counts=settings["sample_counts"],
            client_id=context.client_id,
        )

    def before_step(self, context: TrainingContext) -> None:
        if model_schema(context.model) != self.schema:
            raise ValueError("FedDyn model trainability/schema changed.")
        validate_endpoint(
            context_model(context),
            context_model(context).state_dict(),
            self.global_model_weights,
            self.schema,
        )
        opt = context.state.get("optimizer")
        if not isinstance(opt, torch.optim.SGD) or type(opt) is not torch.optim.SGD:
            raise ValueError("FedDyn supports ordinary SGD only.")
        owned = [p for g in opt.param_groups for p in g["params"]]
        expected = [p for p in context_model(context).parameters() if p.requires_grad]
        if (
            len(owned) != len(expected)
            or len({id(p) for p in owned}) != len(owned)
            or {id(p) for p in owned} != {id(p) for p in expected}
        ):
            raise ValueError("FedDyn optimizer must own every trainable exactly once.")
        rates = []
        for g in opt.param_groups:
            if any(
                g.get(k, 0) != 0 for k in ("momentum", "weight_decay", "dampening")
            ) or any(g.get(k, False) for k in ("nesterov", "maximize")):
                raise ValueError("FedDyn requires plain SGD without momentum/decay.")
            rates.append(finite_real(g["lr"], "learning rate", positive=True))
        if len(set(rates)) != 1 or (self._lr is not None and self._lr != rates[0]):
            raise ValueError("FedDyn requires one fixed local learning rate.")
        if rates[0] != settings_from_config()["lr"]:
            raise ValueError(
                "FedDyn optimizer rate differs from the fixed configuration."
            )
        self._lr = rates[0]
        count = positive_integer(
            context.state.get("feddyn_num_samples"), "realized sample count"
        )
        if (
            context.state.get("num_samples") != count
            or len(context.state["train_loader"].sampler) != count
        ):
            raise ValueError("FedDyn sampler cardinality changed during training.")

    def after_step(self, context: TrainingContext) -> None:
        self.completed_steps += 1

    def on_train_end(self, context: TrainingContext) -> None:
        if self.global_model_weights is None or self.cumulative_grad_vector is None:
            raise ValueError(
                "FedDyn needs a current baseline/history before train end."
            )
        count = positive_integer(
            context.state.get("feddyn_num_samples"), "realized sample count"
        )
        if (
            context.state.get("num_samples") != count
            or len(context.state["train_loader"].sampler) != count
        ):
            raise ValueError(
                "FedDyn sampler cardinality changed before result staging."
            )
        steps = positive_integer(self.completed_steps, "completed optimizer steps")
        y = validate_endpoint(
            context_model(context),
            context_model(context).state_dict(),
            self.global_model_weights,
            self.schema,
        )
        h = OrderedDict(
            (k, v + y[k] - self.global_model_weights[k])
            for k, v in self.cumulative_grad_vector.items()
        )
        h = validate_tensors(h, trainable_reference(context.model), "next history")
        self.result = dict(
            dispatch=copy.deepcopy(self.dispatch),
            endpoint=y,
            history=h,
            num_samples=count,
            completed_steps=steps,
        )

    @property
    def requires_worker_state(self) -> bool:
        return True

    def get_worker_state(self, context: TrainingContext):
        if self.result is None:
            raise ValueError("FedDyn has no completed current worker result.")
        return copy.deepcopy(self.result)

    def load_worker_state(self, state, context: TrainingContext) -> None:
        if not isinstance(state, dict) or set(state) != {
            "dispatch",
            "endpoint",
            "history",
            "num_samples",
            "completed_steps",
        }:
            raise ValueError("FedDyn worker state is missing or malformed.")
        d = state["dispatch"]
        if (
            not isinstance(d, dict)
            or set(d) != {"metadata", "baseline", "schema"}
            or self.dispatch is None
            or d["schema"] != self.dispatch["schema"]
        ):
            raise ValueError("FedDyn worker dispatch/schema mismatch.")
        m, original = d["metadata"], self.dispatch["metadata"]
        if not isinstance(m, dict) or set(m) != set(original):
            raise ValueError("FedDyn worker metadata mismatch.")
        if any(not same_state(m[k], original[k]) for k in original if k != "history"):
            raise ValueError("FedDyn worker has stale/mismatched dispatch identity.")
        x = validate_tensors(
            d["baseline"], self.dispatch["baseline"], "worker baseline"
        )
        h = validate_tensors(m["history"], original["history"], "worker prior history")
        if any(
            not torch.equal(v, self.dispatch["baseline"][k]) for k, v in x.items()
        ) or any(not torch.equal(v, original["history"][k]) for k, v in h.items()):
            raise ValueError("FedDyn worker changed immutable baseline/history.")
        y = validate_endpoint(context.model, state["endpoint"], x, d["schema"])
        if any(
            not torch.equal(v, context_model(context).state_dict()[k].detach().cpu())
            for k, v in y.items()
        ):
            raise ValueError("FedDyn worker model/state pair disagrees.")
        hn = validate_tensors(state["history"], h, "worker next history")
        if any(not torch.equal(v, h[k] + y[k] - x[k]) for k, v in hn.items()):
            raise ValueError("FedDyn worker history disagrees with endpoint.")
        count = positive_integer(state["num_samples"], "worker sample count")
        if (
            original["expected_count_or_null"] is not None
            and count != original["expected_count_or_null"]
        ):
            raise ValueError("FedDyn worker count disagrees with dispatch.")
        self.completed_steps = positive_integer(
            state["completed_steps"], "completed optimizer steps"
        )
        self.result = copy.deepcopy(state)
        context.state["feddyn_num_samples"] = count

    def on_train_result_accepted(self, context: TrainingContext) -> None:
        self.load_worker_state(self.get_worker_state(context), context)
        self.accepted = True

    def on_train_cleanup(self, context: TrainingContext, successful: bool) -> None:
        if not successful:
            self.result = None
            self.accepted = False

    def get_update_payload(self, context: TrainingContext):
        if not self.accepted or self.result is None:
            raise ValueError("FedDyn has no accepted current result.")
        if self.dispatch is None:
            raise ValueError("FedDyn has no active dispatch.")
        m = self.dispatch["metadata"]
        return {
            k: m[k]
            for k in ("version", "run_id", "round", "client_id", "dispatch_token")
        } | {
            "num_samples": self.result["num_samples"],
            "completed_steps": self.result["completed_steps"],
        }

    def read_legacy_history(self, context: TrainingContext):
        """Inspect exact same-client bytes without adoption, creation or writes."""
        if context.client_id == 0:
            return None
        positive_integer(context.client_id, "legacy client ID")
        root = (
            self.save_path
            if self.save_path is not None
            else Config().params["model_path"]
        )
        canonical = Path(
            checkpoint_path(
                root, checkpoint_name("feddyn_grad", context.client_id, suffix=".pth")
            )
        )
        legacy = Path(f"{root}_feddyn_grad_{context.client_id}.pth")
        path = canonical if canonical.is_file() else legacy
        if not path.is_file():
            return None
        values = torch.load(path, weights_only=True, map_location="cpu")
        q = trainable_reference(context.model)
        frozen = {
            n: p
            for n, p in context_model(context).named_parameters()
            if not p.requires_grad
        }
        if not isinstance(values, Mapping) or set(values) - set(q) - set(frozen):
            raise ValueError("FedDyn legacy history has unrecognized keys.")
        for name in set(values) & set(frozen):
            validate_tensors(
                {name: values[name]}, {name: frozen[name]}, "legacy frozen extra"
            )
        return validate_tensors(
            {k: v for k, v in values.items() if k in q}, q, "legacy history"
        )

    def teardown(self, context: TrainingContext) -> None:
        self.dispatch = self.result = None
        self.global_model_weights = self.cumulative_grad_vector = None
        self.accepted = False
        for key in list(context.state):
            if key.startswith("feddyn_"):
                context.state.pop(key)


class FedDynLossStrategyFromConfig(FedDynLossStrategy):
    """Resolve alpha precedence and configured population weighting."""

    def __init__(
        self, base_loss_fn: Callable | None = None, adaptive_alpha: bool | None = None
    ):
        c = Config()
        mode = getattr(c.algorithm, "feddyn_weighting", "uniform")
        if mode not in ("uniform", "sample"):
            raise ValueError("FedDyn weighting must be uniform or sample.")
        resolved = mode == "sample"
        if adaptive_alpha is not None and (
            type(adaptive_alpha) is not bool or adaptive_alpha != resolved
        ):
            raise ValueError(
                "FedDyn adaptive_alpha conflicts with configured weighting."
            )
        alpha = getattr(
            c.algorithm, "alpha_coef", getattr(c.algorithm, "feddyn_alpha", 0.01)
        )
        super().__init__(
            alpha=alpha, base_loss_fn=base_loss_fn, adaptive_alpha=resolved
        )
