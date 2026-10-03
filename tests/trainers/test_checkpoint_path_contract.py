"""Real checkpoint writers/readers and logical-name isolation."""

import copy
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.utils.checkpoint_paths import checkpoint_component, checkpoint_name
from tests.integration.utils import build_minimal_config, configure_environment


@pytest.mark.parametrize(
    "name", ["toy", "org/model", "org_model", "../model", "~6f72672f6d6f64656c"]
)
def test_default_worker_test_and_cleanup_checkpoint_names(tmp_path, name):
    config = build_minimal_config(model_name=name)
    config["trainer"].update(batch_size=2, epochs=1, max_concurrency=1)
    with configure_environment(config, runtime_root=tmp_path):
        trainer = ComposableTrainer(model=torch.nn.Linear(2, 2))
        trainer.set_client_id(7)
        trainer.device = trainer.context.device = torch.device("cpu")
        root = Path(Config.params["model_path"])
        trainer.run_history.update_metric("custom", 12)
        expected = copy.deepcopy(trainer.model.state_dict())
        trainer.save_model()
        trainer.model.weight.data.zero_()
        trainer.load_model()
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, expected[key])
        data = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
        run = {**config["trainer"], "run_id": Config.params["run_id"]}
        trainer.train_process(run, data, [0, 1])
        trainer.test_process(run, data)
        trainer.context.state.clear()
        trainer._load_test_state(trainer._test_state_filename(run["run_id"]))
        worker = checkpoint_name(name, 7, run["run_id"], suffix=".safetensors")
        assert (root / worker).is_file()
        assert (root / (worker + ".pkl")).is_file()
        assert (root / trainer._test_accuracy_filename(run["run_id"])).is_file()
        assert all(file.parent == root for file in root.iterdir())
        unrelated = root / checkpoint_name(
            name, 8, run["run_id"], suffix=".safetensors"
        )
        unrelated.write_bytes(b"other client")
        trainer.pause_training()
        assert not (root / worker).exists()
        assert not (root / (worker + ".pkl")).exists()
        assert not (root / trainer._test_accuracy_filename(run["run_id"])).exists()
        assert not (root / trainer._test_state_filename(run["run_id"])).exists()
        assert unrelated.is_file()
        assert (root / checkpoint_name(name, suffix=".safetensors")).is_file()


def test_logical_name_codec_has_no_slash_underscore_or_reserved_collision():
    names = ["org/model", "org_model", "~6f72672f6d6f64656c", "../toy", "", ".."]
    assert len({checkpoint_component(name) for name in names}) == len(names)
    assert checkpoint_name("lenet5", 2, "run", suffix=".safetensors") == (
        "lenet5_2_run.safetensors"
    )


@pytest.mark.parametrize(
    "name", ["org/" + "model" * 80, "ordinary_" + "model" * 80, "x" * 240]
)
def test_long_logical_model_name_default_save_load_stays_in_filename_limit(
    tmp_path, name
):
    with configure_environment(
        build_minimal_config(model_name=name), runtime_root=tmp_path
    ):
        trainer = ComposableTrainer(model=torch.nn.Linear(2, 2))
        expected = copy.deepcopy(trainer.model.state_dict())
        trainer.save_model()
        trainer.model.weight.data.zero_()
        trainer.load_model()
        torch.testing.assert_close(trainer.model.weight, expected["weight"])
        assert len(checkpoint_name(name, suffix=".safetensors").encode()) <= 255


def test_explicit_location_and_relative_filename_containment(tmp_path):
    config = build_minimal_config()
    with configure_environment(config, runtime_root=tmp_path):
        trainer = ComposableTrainer(model=torch.nn.Linear(2, 2))
        custom = tmp_path / "custom"
        trainer.save_model("nested/model.safetensors", location=custom)
        trainer.load_model("nested/model.safetensors", location=custom)
        for filename in (
            "../outside.safetensors",
            str(tmp_path / "absolute.safetensors"),
        ):
            with pytest.raises(ValueError):
                trainer.save_model(filename, location=custom)
            with pytest.raises(ValueError):
                trainer.load_model(filename, location=custom)
        (custom / "escape").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError):
            trainer.save_model("escape/outside.safetensors", location=custom)


def test_urgent_actual_snapshots_numerical_order_cutoff_architecture_and_state(
    tmp_path,
):
    config = build_minimal_config(model_name="custom")
    with configure_environment(config, runtime_root=tmp_path):
        trainer = ComposableTrainer(model=torch.nn.Linear(2, 2))
        trainer.set_client_id(7)
        trainer.device = trainer.context.device = torch.device("cpu")
        trainer.model.train()
        for client, epoch, time, value in [
            (7, 9, 9.0, 9.0),
            (7, 10, 10.0, 10.0),
            (8, 20, 10.5, 20.0),
        ]:
            trainer.model.weight.data.fill_(value)
            trainer.save_model(f"{client}_{epoch}_{time}.safetensors")
        current = copy.deepcopy(trainer.model.state_dict())
        for cutoff, expected in [(11.0, 10.0), (10.0, 9.0)]:
            historical = trainer.obtain_model_at_time(7, cutoff)
            assert isinstance(historical, torch.nn.Linear)
            assert torch.all(historical.weight == expected)
            assert historical.training == trainer.model.training
            assert historical.weight.device == trainer.model.weight.device
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, current[key])
        with pytest.raises(ValueError, match="Cannot find"):
            trainer.obtain_model_at_time(7, 9.0)
        with pytest.raises(ValueError, match="Cannot find"):
            trainer.obtain_model_at_time(99, 100.0)


def test_real_spawned_worker_train_test_and_history_with_slash_name(tmp_path):
    """The parent consumes the child's actual model and testing artifacts."""
    config = build_minimal_config(model_name="org/model")
    config["trainer"].update(batch_size=2, epochs=1, max_concurrency=1)
    with configure_environment(config, runtime_root=tmp_path):
        torch.manual_seed(31)
        trainer = ComposableTrainer(model=torch.nn.Linear(2, 2))
        trainer.set_client_id(7)
        trainer.device = trainer.context.device = torch.device("cpu")
        before = copy.deepcopy(trainer.model.state_dict())
        data = TensorDataset(torch.eye(2).repeat(2, 1), torch.tensor([0, 1, 0, 1]))
        assert trainer.train(data, [0, 1, 2, 3]) >= 0
        assert any(
            not torch.equal(value, before[key])
            for key, value in trainer.model.state_dict().items()
        )
        assert trainer.run_history.get_metric_values("train_loss")
        assert 0.0 <= trainer.test(data) <= 1.0


def test_split_gradient_auxiliary_writer_reader_uses_same_name(tmp_path):
    from plato.trainers.split_learning import Trainer

    config = build_minimal_config(model_name="org/model", trainer_type="split_learning")
    with configure_environment(config, runtime_root=tmp_path):
        trainer = Trainer(model=torch.nn.Linear(2, 2))
        trainer.cut_layer_grad = [torch.tensor([[1.0, 2.0]])]
        trainer.save_gradients(config["trainer"])
        restored = trainer.get_gradients()
        torch.testing.assert_close(restored[0], trainer.cut_layer_grad[0])


def test_history_sidecar_symlinks_are_checked_before_weight_write_or_load(tmp_path):
    with configure_environment(build_minimal_config(), runtime_root=tmp_path):
        trainer = ComposableTrainer(model=torch.nn.Linear(2, 2))
        custom = tmp_path / "custom"
        custom.mkdir()
        outside = tmp_path / "outside.pkl"
        outside.write_bytes(b"retained outside data")
        history = custom / "weights.safetensors.pkl"
        history.symlink_to(outside)
        with pytest.raises(ValueError, match="within"):
            trainer.save_model("weights.safetensors", custom)
        assert outside.read_bytes() == b"retained outside data"
        assert not (custom / "weights.safetensors").exists()
        history.unlink()
        trainer.save_model("weights.safetensors", custom)
        history.unlink()
        history.symlink_to(outside)
        trainer.model.weight.data.zero_()
        with pytest.raises(ValueError, match="within"):
            trainer.load_model("weights.safetensors", custom)
        assert torch.all(trainer.model.weight == 0)
