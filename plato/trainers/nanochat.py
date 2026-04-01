"""
Trainer wiring for Nanochat models within the composable trainer framework.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Sequence, overload

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Nanochat trainer requires PyTorch. "
        "Install torch via the project's optional dependencies."
    ) from exc

from plato.config import Config
from plato.datasources.nanochat import NanochatStreamingDataset
from plato.evaluators.nanochat_core import (
    NANOCHAT_CORE_RESULTS_KEY,
    NanochatCoreEvaluator,
    run_core_evaluation,
)
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    DataLoaderStrategy,
    OptimizerStrategy,
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)
from plato.trainers.strategies.data_loader import DefaultDataLoaderStrategy
from plato.utils.third_party import ThirdPartyImportError, ensure_nanochat_importable


def _first_element_collate(batch: Sequence[Any]) -> Any:
    """Return the first (and only) element from a DataLoader batch."""
    if not batch:
        raise ValueError("Received empty batch from Nanochat dataset.")
    return batch[0]


class NanochatDataLoaderStrategy(DataLoaderStrategy):
    """Use identity collation for pre-batched Nanochat streaming datasets."""

    def __init__(self):
        self._fallback = DefaultDataLoaderStrategy()

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        if isinstance(trainset, NanochatStreamingDataset):
            return torch.utils.data.DataLoader(
                trainset,
                batch_size=1,
                shuffle=False,
                sampler=None,
                num_workers=0,
                collate_fn=_first_element_collate,
            )
        return self._fallback.create_train_loader(
            trainset, sampler, batch_size, context
        )


class NanochatTrainingStepStrategy(TrainingStepStrategy):
    """Call Nanochat's integrated loss computation during training."""

    def __init__(self, loss_reduction: str = "mean"):
        self.loss_reduction = loss_reduction

    def training_step(
        self,
        model: torch.nn.Module,
        optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion,
        context: TrainingContext,
    ) -> torch.Tensor:
        optimizer.zero_grad()
        if labels is not None:
            loss = model(examples, targets=labels, loss_reduction=self.loss_reduction)
        else:
            outputs = model(examples)
            loss = loss_criterion(outputs, labels)

        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                "Nanochat model forward pass must return a torch.Tensor loss."
            )

        loss.backward()
        optimizer.step()
        context.state["optimizer_step_completed"] = True
        return loss.detach()


class _OptimizerBundle(torch.optim.Optimizer):
    """Bundle multiple optimizers under a single interface."""

    optimizers: List[torch.optim.Optimizer]

    def __init__(self, optimizers: List[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        param_groups: list[dict[str, Any]] = []
        for optimizer in self.optimizers:
            param_groups.extend(getattr(optimizer, "param_groups", []))
        super().__init__(param_groups, {})

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in self.optimizers:
            try:
                optimizer.zero_grad(set_to_none=set_to_none)
            except TypeError:
                optimizer.zero_grad()

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @overload
    def step(self, closure: None = ...) -> None: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optimizer, payload in zip(
            self.optimizers,
            state_dict.get("optimizers", []),
            strict=False,
        ):
            optimizer.load_state_dict(payload)

    def params_state_update(self) -> None:
        for optimizer in self.optimizers:
            hook = getattr(optimizer, "params_state_update", None)
            if callable(hook):
                hook()


class NanochatOptimizerStrategy(OptimizerStrategy):
    """Adapter around nanochat.gpt.GPT.setup_optimizers."""

    def __init__(
        self,
        *,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
    ):
        self.unembedding_lr = unembedding_lr
        self.embedding_lr = embedding_lr
        self.matrix_lr = matrix_lr
        self.weight_decay = weight_decay

    def create_optimizer(
        self, model: torch.nn.Module, context: TrainingContext
    ) -> _OptimizerBundle:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Nanochat optimizer strategy requires a torch.nn.Module.")

        setup_fn = getattr(model, "setup_optimizers", None)
        if not callable(setup_fn):
            raise AttributeError(
                "Nanochat model is expected to expose setup_optimizers()."
            )

        optimizers = setup_fn(
            unembedding_lr=self.unembedding_lr,
            embedding_lr=self.embedding_lr,
            matrix_lr=self.matrix_lr,
            weight_decay=self.weight_decay,
        )
        if not isinstance(optimizers, Iterable):
            raise TypeError("setup_optimizers() must return an iterable of optimizers.")

        optimizer_candidates = list(optimizers)
        optimizer_list: list[torch.optim.Optimizer] = []
        for optimizer in optimizer_candidates:
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError(
                    "setup_optimizers() must yield torch.optim.Optimizer instances."
                )
            optimizer_list.append(optimizer)
        if not optimizer_list:
            raise ValueError("setup_optimizers() returned an empty optimizer list.")
        return _OptimizerBundle(optimizer_list)


class NanochatTestingStrategy(TestingStrategy):
    """Compute average token loss over the validation iterator."""

    def __init__(self, reduction: str = "sum"):
        self.reduction = reduction

    def test_model(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        testset,
        sampler,
        context: TrainingContext,
    ) -> float:
        if not isinstance(testset, NanochatStreamingDataset):
            raise TypeError(
                "NanochatTestingStrategy expects a NanochatStreamingDataset instance."
            )

        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in testset:
                inputs = inputs.to(context.device)
                targets = targets.to(context.device)
                loss = model(inputs, targets=targets, loss_reduction=self.reduction)
                total_loss += float(loss.item())
                total_tokens += targets.numel()

        model.train()
        if total_tokens == 0:
            return float("nan")
        return total_loss / total_tokens


class NanochatCoreTestingStrategy(TestingStrategy):
    """Evaluate the CORE benchmark and return the aggregate metric."""

    def __init__(self, bundle_dir: str | None = None, max_per_task: int = -1):
        self.bundle_dir = bundle_dir
        self.max_per_task = max_per_task

    def test_model(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        testset,
        sampler,
        context: TrainingContext,
    ) -> float:
        device = context.device or next(model.parameters()).device
        tokenizer = getattr(model, "nanochat_tokenizer", None)
        results = run_core_evaluation(
            model,
            tokenizer=tokenizer,
            bundle_dir=self.bundle_dir,
            max_per_task=self.max_per_task,
            device=device,
        )
        context.state[NANOCHAT_CORE_RESULTS_KEY] = results
        return float(results["core_metric"])


class Trainer(ComposableTrainer):
    """Composable trainer specialised for Nanochat workloads."""

    @staticmethod
    def _resolve_evaluation_components(
        evaluation_cfg: Any | None,
    ) -> tuple[TestingStrategy, NanochatCoreEvaluator | None]:
        evaluation_type = (
            getattr(evaluation_cfg, "type", "").lower() if evaluation_cfg else ""
        )
        if evaluation_type == "nanochat_core":
            max_per_task = getattr(evaluation_cfg, "max_per_task", -1)
            max_per_task_value = -1 if max_per_task is None else int(max_per_task)
            return (
                NanochatCoreTestingStrategy(
                    bundle_dir=getattr(evaluation_cfg, "bundle_dir", None),
                    max_per_task=max_per_task_value,
                ),
                NanochatCoreEvaluator(evaluation_cfg),
            )

        return NanochatTestingStrategy(), None

    def _refresh_evaluation_mode(self) -> None:
        evaluation_cfg = getattr(Config(), "evaluation", None)
        testing_strategy, evaluator_override = self._resolve_evaluation_components(
            evaluation_cfg
        )
        self.testing_strategy = testing_strategy
        self._configured_evaluator_override = evaluator_override

        if evaluator_override is None:
            self.context.state.pop(NANOCHAT_CORE_RESULTS_KEY, None)

    def __init__(
        self,
        model=None,
        callbacks=None,
        *,
        optimizer_params: dict[str, Any] | None = None,
        loss_reduction: str = "mean",
    ):
        try:
            ensure_nanochat_importable()
        except ThirdPartyImportError as exc:  # pragma: no cover - defensive branch
            raise ImportError(
                "Nanochat trainer requires the external/nanochat submodule. "
                "Run `git submodule update --init --recursive`."
            ) from exc

        optimizer_strategy = NanochatOptimizerStrategy(
            **(optimizer_params or {}),
        )
        training_step_strategy = NanochatTrainingStepStrategy(
            loss_reduction=loss_reduction
        )
        data_loader_strategy = NanochatDataLoaderStrategy()
        self._configured_evaluator_override = None

        evaluation_cfg = getattr(Config(), "evaluation", None)
        testing_strategy, self._configured_evaluator_override = (
            self._resolve_evaluation_components(evaluation_cfg)
        )

        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=None,
            optimizer_strategy=optimizer_strategy,
            training_step_strategy=training_step_strategy,
            lr_scheduler_strategy=None,
            model_update_strategy=None,
            data_loader_strategy=data_loader_strategy,
            testing_strategy=testing_strategy,
        )

    def test_model(self, config, testset, sampler=None, **kwargs):
        self._refresh_evaluation_mode()
        return super().test_model(config, testset, sampler=sampler, **kwargs)
