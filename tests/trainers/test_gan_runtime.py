"""Actual GAN registration, training, exchange and paired checkpoints."""

import copy
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.algorithms.fedavg_gan import Algorithm
from plato.models import dcgan
from plato.trainers.gan import GANTestingStrategy, Trainer
from plato.trainers.strategies.base import TrainingContext
from tests.integration.utils import build_minimal_config, configure_environment


def test_actual_registered_dcgan_training_changes_both_networks(tmp_path):
    from plato.trainers import registry

    config = build_minimal_config(model_name="dcgan", trainer_type="gan")
    config["trainer"].update(batch_size=2, epochs=1)
    with configure_environment(config, runtime_root=tmp_path):
        torch.manual_seed(23)
        trainer = registry.get()
        assert isinstance(trainer.model, torch.nn.Module)
        trainer.device = "cpu"
        trainer.context.device = torch.device("cpu")
        assert trainer.model is not None
        before_g = copy.deepcopy(trainer.generator.state_dict())
        before_d = copy.deepcopy(trainer.discriminator.state_dict())
        data = TensorDataset(torch.rand(2, 3, 64, 64) * 2 - 1, torch.zeros(2))
        trainer.train_model({**config["trainer"], "run_id": "gan"}, data, [0, 1])
        for network, before in (
            (trainer.generator, before_g),
            (trainer.discriminator, before_d),
        ):
            assert any(
                not torch.equal(value, before[key])
                for key, value in network.state_dict().items()
            )
            assert all(torch.isfinite(p).all() for p in network.parameters())
        trainer.model.eval()
        assert not trainer.generator.training and not trainer.discriminator.training
        trainer.model.train()
        assert trainer.generator.training and trainer.discriminator.training


def test_gan_exchange_owns_snapshot_and_preserves_model_device(tmp_path):
    config = build_minimal_config(model_name="dcgan", trainer_type="gan")
    with configure_environment(config, runtime_root=tmp_path):
        trainer = Trainer(model=dcgan.Model())
        algorithm = Algorithm(trainer)
        payload = algorithm.extract_weights()
        first_key = next(iter(payload[0]))
        original = payload[0][first_key].clone()
        with torch.no_grad():
            trainer.generator.state_dict()[first_key].add_(1)
        torch.testing.assert_close(payload[0][first_key], original)
        algorithm.load_weights(payload)
        torch.testing.assert_close(trainer.generator.state_dict()[first_key], original)


def test_gan_slash_names_and_explicit_paired_checkpoints(tmp_path):
    config = build_minimal_config(model_name="org/model", trainer_type="gan")
    with configure_environment(config, runtime_root=tmp_path):
        trainer = Trainer(model=dcgan.Model())
        custom = tmp_path / "gan"
        for filename in (None, "pair.pth", "worker.safetensors"):
            before = (
                copy.deepcopy(trainer.generator.state_dict()),
                copy.deepcopy(trainer.discriminator.state_dict()),
            )
            trainer.save_model(filename, location=custom)
            with torch.no_grad():
                next(trainer.generator.parameters()).add_(1)
                next(trainer.discriminator.parameters()).add_(1)
            trainer.load_model(filename, location=custom)
            for network, expected in zip(
                (trainer.generator, trainer.discriminator), before
            ):
                for key, value in network.state_dict().items():
                    torch.testing.assert_close(value, expected[key])
        assert all(path.parent == custom for path in custom.iterdir())
        assert (custom / "Generator_pair.pth").is_file()
        assert (custom / "Discriminator_pair.pth").is_file()
        assert (custom / "worker.safetensors").is_file()
        # Older GAN workers used Torch-format pairs with this suffix.
        expected = copy.deepcopy(trainer.generator.state_dict())
        torch.save(expected, custom / "Generator_legacy.safetensors")
        torch.save(
            trainer.discriminator.state_dict(),
            custom / "Discriminator_legacy.safetensors",
        )
        next(trainer.generator.parameters()).data.zero_()
        trainer.load_model("legacy.safetensors", location=custom)
        for key, value in trainer.generator.state_dict().items():
            torch.testing.assert_close(value, expected[key])


def test_gan_worker_history_urgent_snapshots_and_scoped_cleanup(tmp_path):
    from plato.config import Config
    from plato.utils.checkpoint_paths import checkpoint_name

    config = build_minimal_config(model_name="org/model", trainer_type="gan")
    config["trainer"].update(batch_size=2, epochs=1, max_concurrency=1)
    with configure_environment(config, runtime_root=tmp_path):
        trainer = Trainer(model=dcgan.Model())
        trainer.set_client_id(7)
        trainer.run_history.update_metric("gan_metric", 9.0)
        worker = checkpoint_name(
            "org/model", 7, Config.params["run_id"], suffix=".safetensors"
        )
        trainer.save_model(worker)
        trainer.run_history.reset()
        trainer.load_model(worker)
        assert trainer.run_history.get_latest_metric("gan_metric") == 9.0
        for epoch in (9, 10):
            next(trainer.generator.parameters()).data.fill_(float(epoch))
            trainer.save_model(f"7_{epoch}_{float(epoch)}.safetensors")
        historical = trainer.obtain_model_at_time(7, 11.0)
        assert next(historical.generator.parameters()).flatten()[0].item() == 10.0
        unrelated = checkpoint_name(
            "org/model", 8, Config.params["run_id"], suffix=".safetensors"
        )
        trainer.save_model(unrelated)
        trainer.pause_training()
        root = Path(Config.params["model_path"])
        assert not (root / worker).exists() and not (root / (worker + ".pkl")).exists()
        assert (root / unrelated).is_file() and (
            root / "7_10_10.0.safetensors"
        ).is_file()


def test_gan_deltas_and_updates_preserve_bool_and_integer_buffer_contract(tmp_path):
    class BufferedGAN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.generator = torch.nn.Linear(2, 2)
            self.discriminator = torch.nn.Linear(2, 2)
            for network in (self.generator, self.discriminator):
                network.register_buffer("flag", torch.tensor([False]))
                network.register_buffer("count", torch.tensor([2], dtype=torch.int64))

    with configure_environment(
        build_minimal_config(trainer_type="gan"), runtime_root=tmp_path
    ):
        trainer = Trainer(model=BufferedGAN())
        algorithm = Algorithm(trainer)
        baseline = algorithm.extract_weights()
        changed = copy.deepcopy(baseline)
        for state in changed:
            state["flag"].fill_(True)
            state["count"].fill_(5)
        (delta,) = algorithm.compute_weight_deltas(baseline, [changed])
        updated = algorithm.update_weights(delta)
        for state in updated:
            assert state["flag"].dtype == torch.bool and state["flag"].item() is True
            assert state["count"].dtype == torch.int64 and state["count"].item() == 5
        algorithm.load_weights(updated)
        assert trainer.generator.flag.item() is True
        assert trainer.discriminator.count.item() == 5


def test_gan_fid_actual_partition_tail_padding_and_scalar_covariance_reference():
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, noise):
            self.batch_sizes.append(len(noise))
            return torch.zeros(len(noise), 3, 32, 80)

    class FeatureExtractor(torch.nn.Module):
        def forward(self, images):
            assert images.shape[-2] >= 75 and images.shape[-1] >= 75
            # Central pixel survives padding; real features are [1,3,5], with
            # mean 3 and unbiased variance 4. Generated features are zero.
            value = images[:, 0, images.shape[-2] // 2, images.shape[-1] // 2]
            return torch.stack((value, torch.zeros_like(value)), dim=1)

    class GANModel(torch.nn.Module):
        nz: int = 2

        def __init__(self):
            super().__init__()
            self.generator = Generator()

    model = GANModel()
    strategy = GANTestingStrategy.__new__(GANTestingStrategy)
    strategy.inception_model = FeatureExtractor()
    images = torch.arange(1, 6).view(5, 1, 1, 1).expand(5, 3, 32, 80).float()
    data = TensorDataset(images, torch.zeros(5))
    context = TrainingContext()
    context.device = torch.device("cpu")
    sampler = type("Partition", (), {"get": lambda self: [0, 2, 4]})()
    score = strategy.test_model(model, {"batch_size": 2}, data, sampler, context)
    epsilon = 1e-6
    expected = 9 + 4 + 2 * epsilon - 2 * math.sqrt((4 + epsilon) * epsilon)
    assert score == pytest.approx(expected)
    assert model.generator.batch_sizes == [2, 1]
    for partition in ([], [0]):
        with pytest.raises(ValueError, match="at least two"):
            strategy.test_model(model, {"batch_size": 2}, data, partition, context)


def test_gan_explicit_normalized_paths_keep_networks_distinct_and_validate_first(
    tmp_path,
):
    config = build_minimal_config(trainer_type="gan")
    with configure_environment(config, runtime_root=tmp_path):
        trainer = Trainer(model=dcgan.Model())
        expected = (
            copy.deepcopy(trainer.generator.state_dict()),
            copy.deepcopy(trainer.discriminator.state_dict()),
        )
        custom = tmp_path / "pairs"
        for filename in ("nested/pair.pth", "nested/../normalized.pth"):
            trainer.run_history.update_metric("marker", 7)
            trainer.save_model(filename, custom)
            next(trainer.generator.parameters()).data.zero_()
            next(trainer.discriminator.parameters()).data.zero_()
            trainer.run_history.reset()
            trainer.load_model(filename, custom)
            for network, state in zip(
                (trainer.generator, trainer.discriminator), expected
            ):
                for name, value in network.state_dict().items():
                    torch.testing.assert_close(value, state[name])
            assert trainer.run_history.get_latest_metric("marker") == 7
        assert (custom / "Generator_nested/pair.pth").is_file()
        assert (custom / "Discriminator_nested/pair.pth").is_file()
        assert (custom / "Generator_normalized.pth").is_file()
        assert (custom / "Discriminator_normalized.pth").is_file()
        for filename in ("../outside.pth", str(tmp_path / "absolute.pth")):
            before = {path for path in custom.rglob("*") if path.is_file()}
            with pytest.raises(ValueError):
                trainer.save_model(filename, custom)
            assert {path for path in custom.rglob("*") if path.is_file()} == before
