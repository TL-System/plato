import importlib.util

import pytest
import torch
from torch.utils.data import SubsetRandomSampler, TensorDataset

if importlib.util.find_spec("opacus") is None:
    pytest.skip("base profile optional dp: opacus not installed", allow_module_level=True)

from plato.trainers.diff_privacy import DPDataLoaderStrategy
from plato.trainers.strategies.base import TrainingContext


def test_actual_spawned_dp_rounds_return_accounting_and_isolate_clients(tmp_path):
    from plato.trainers.diff_privacy import Trainer
    from tests.integration.utils import build_minimal_config, configure_environment

    config = build_minimal_config(trainer_type="diff_privacy")
    config["trainer"].update(
        batch_size=4, epochs=1, max_physical_batch_size=2, max_concurrency=1,
    )
    with configure_environment(config, runtime_root=tmp_path):
        torch.manual_seed(7)
        trainer = Trainer(model=torch.nn.Linear(2, 2))
        trainer.device = trainer.context.device = torch.device("cpu")
        trainer.set_client_id(1)
        dataset = TensorDataset(torch.randn(16, 2), torch.arange(16) % 2)

        def steps():
            engine = trainer.optimizer_strategy.privacy_engine
            assert engine is not None
            return sum(entry[2] for entry in engine.accountant.history)

        for expected_steps in (4, 8):
            before = {
                key: value.clone() for key, value in trainer.model.state_dict().items()
            }
            trainer.train(dataset, list(range(16)))
            assert steps() == expected_steps
            assert not hasattr(trainer.model, "autograd_grad_sample_hooks")
            assert any(
                not torch.equal(value, before[key])
                for key, value in trainer.model.state_dict().items()
            )
            trainer.save_model()
            trainer.load_model()
            assert set(trainer.model.state_dict()) == {"weight", "bias"}
        trainer.set_client_id(2)
        assert trainer.optimizer_strategy.privacy_engine is None
        trainer.train(dataset, list(range(16)))
        assert steps() == 4
        trainer.set_client_id(1)
        assert steps() == 8


class _FakePlatoSampler:
    """Minimal stub to mimic Plato sampler behaviour with subset indices."""

    def __init__(self, indices):
        self.subset_indices = indices

    def get(self):
        return SubsetRandomSampler(self.subset_indices)


def _collect_dataset_indices(loader):
    """Utility to gather indices from batches for assertions."""
    collected = []
    for values, _ in loader:
        collected.extend(values.tolist())
    return sorted(collected)


def test_dp_strategy_handles_plato_sampler_get():
    """DP data loader should resolve Plato sampler objects into subset indices."""
    dataset = TensorDataset(torch.arange(10), torch.arange(10))
    sampler = _FakePlatoSampler([1, 3, 5, 7])
    context = TrainingContext()

    loader = DPDataLoaderStrategy().create_train_loader(
        dataset, sampler, batch_size=2, context=context
    )

    assert _collect_dataset_indices(loader) == [1, 3, 5, 7]


def test_dp_strategy_handles_torch_sampler_directly():
    """DP data loader should accept native PyTorch samplers."""
    dataset = TensorDataset(torch.arange(8), torch.arange(8))
    torch_sampler = SubsetRandomSampler([0, 2, 4, 6])
    context = TrainingContext()

    loader = DPDataLoaderStrategy().create_train_loader(
        dataset, torch_sampler, batch_size=2, context=context
    )

    assert _collect_dataset_indices(loader) == [0, 2, 4, 6]


def test_actual_dp_consecutive_training_exchange_checkpoint_and_accounting(tmp_path):
    """Real Opacus rounds retain accounting and remove wrapper hooks/keys."""
    from plato.algorithms.fedavg import Algorithm
    from plato.trainers.diff_privacy import Trainer
    from tests.integration.utils import build_minimal_config, configure_environment

    config = build_minimal_config(trainer_type="diff_privacy")
    config["trainer"].update(batch_size=4, epochs=1, max_physical_batch_size=2)
    with configure_environment(config, runtime_root=tmp_path):
        torch.manual_seed(17)
        trainer = Trainer(model=torch.nn.Linear(2, 2))
        trainer.device = trainer.context.device = torch.device("cpu")
        dataset = TensorDataset(torch.randn(16, 2), torch.arange(16) % 2)
        run = {**config["trainer"], "run_id": "dp"}
        initial = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        engine = None
        for round_id in (1, 2):
            before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            trainer.train_model(run, dataset, list(range(16)))
            assert trainer.context.model is trainer.model
            assert not hasattr(trainer.model, "autograd_grad_sample_hooks")
            current = trainer.model.state_dict()
            assert set(current) == set(initial)
            assert any(not torch.equal(before[k], current[k]) for k in current)
            actual_engine = trainer.optimizer_strategy.privacy_engine
            if engine is not None:
                assert actual_engine is engine
            engine = actual_engine
            assert sum(entry[2] for entry in engine.accountant.history) == round_id * 4
            exchanged = Algorithm(trainer).extract_weights()
            assert set(exchanged) == set(initial)
            trainer.save_model()
            with torch.no_grad():
                trainer.model.weight.add_(100)
            trainer.load_model()
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, exchanged[key])


def test_actual_dp_physical_batches_only_report_real_updates_and_cleanup(tmp_path):
    """Observe the underlying optimizer and accountant under memory splitting."""
    from opacus.optimizers import DPOptimizer

    from plato.callbacks.trainer import TrainerCallback
    from plato.trainers.diff_privacy import Trainer
    from tests.integration.utils import build_minimal_config, configure_environment

    class Observe(TrainerCallback):
        def __init__(self):
            self.physical_batches = 0
            self.actual_updates = 0
            self.reported_updates = 0
            self.samples = 0
            self.interrupt = False

        def on_train_epoch_start(self, trainer, config, **kwargs):
            assert isinstance(trainer.optimizer, DPOptimizer)
            base_optimizer = trainer.optimizer.original_optimizer
            original = base_optimizer.step

            def observed_step(*args, **kwargs):
                self.actual_updates += 1
                return original(*args, **kwargs)

            base_optimizer.step = observed_step

        def on_train_step_start(self, trainer, config, **kwargs):
            self.physical_batches += 1

        def on_train_step_end(self, trainer, config, **kwargs):
            assert trainer.context.state["optimizer_step_completed"] is True
            assert trainer.optimizer._is_last_step_skipped is False
            self.reported_updates += 1
            if self.interrupt:
                raise RuntimeError("interrupt DP after real update")

    config = build_minimal_config(trainer_type="diff_privacy")
    config["trainer"].update(batch_size=4, epochs=1, max_physical_batch_size=2)
    with configure_environment(config, runtime_root=tmp_path):
        torch.manual_seed(17)
        observer = Observe()
        trainer = Trainer(model=torch.nn.Linear(2, 2), callbacks=[observer])
        trainer.device = trainer.context.device = torch.device("cpu")
        data = TensorDataset(torch.randn(16, 2), torch.arange(16) % 2)
        run = {**config["trainer"], "run_id": "dp-count"}
        trainer.train_model(run, data, list(range(16)))
        history = trainer.optimizer_strategy.privacy_engine.accountant.history
        assert observer.physical_batches > 4
        assert observer.actual_updates == observer.reported_updates == 4
        assert sum(entry[2] for entry in history) == 4
        observer.interrupt = True
        with pytest.raises(RuntimeError, match="interrupt DP"):
            trainer.train_model(run, data, list(range(16)))
        assert not hasattr(trainer.model, "autograd_grad_sample_hooks")
        assert trainer.context.model is trainer.model
        assert sum(entry[2] for entry in history) == 5
        observer.interrupt = False
        trainer.train_model(run, data, list(range(16)))
        assert sum(entry[2] for entry in history) == 9


def test_dp_memory_splitting_matches_logical_batch_parameter_reference():
    """All fixed samples contribute once, with real clipping and accountant steps."""
    import copy

    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from torch.utils.data import DataLoader

    from plato.trainers.diff_privacy import DPTrainingStepStrategy

    torch.manual_seed(29)
    reference = torch.nn.Linear(2, 2)
    split_model = copy.deepcopy(reference)
    dataset = TensorDataset(torch.randn(8, 2), torch.arange(8) % 2)
    context = TrainingContext()
    context.device = torch.device("cpu")
    results = []
    for model, physical_size in ((reference, 4), (split_model, 2)):
        engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        private_model, optimizer, loader = engine.make_private(
            module=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            data_loader=DataLoader(dataset, batch_size=4), noise_multiplier=0.,
            max_grad_norm=1., poisson_sampling=False,
        )
        flags = []
        observed_samples = 0
        with BatchMemoryManager(data_loader=loader, max_physical_batch_size=physical_size,
                                optimizer=optimizer) as batches:
            for examples, labels in batches:
                DPTrainingStepStrategy().training_step(
                    private_model, optimizer, examples, labels,
                    torch.nn.functional.cross_entropy, context,
                )
                flags.append(context.state["optimizer_step_completed"])
                observed_samples += len(labels)
        assert observed_samples == 8
        assert sum(entry[2] for entry in engine.accountant.history) == 2
        results.append(flags)
        private_model.to_standard_module()
    assert results == [[True, True], [False, True, False, True]]
    for actual, expected in zip(split_model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
