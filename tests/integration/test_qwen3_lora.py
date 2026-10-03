"""Network-free random native Qwen3 coverage; not pretrained qualification."""

import json

import pytest
import torch

from tests.integration.utils import configure_environment
from tests.test_utils.qwen3 import (
    create_tiny_qwen3,
    reference_config,
    validate_runtime,
)


@pytest.mark.integration
def test_offline_qwen3_peft_two_client_update_and_checkpoint(tmp_path):
    model_directory = create_tiny_qwen3(tmp_path / "tiny-qwen3")
    config = reference_config(model_directory)
    with configure_environment(config):
        result = validate_runtime(tmp_path / "checkpoints")
    assert result["model_class"] == "Qwen3ForCausalLM"
    assert result["client_sample_counts"] == [2, 3]
    assert result["trainable_parameters"] == 448
    assert result["before_adapter_sha256"] != result["after_adapter_sha256"]
    metadata = json.loads((tmp_path / "checkpoints/global.safetensors.hf").read_text())
    assert metadata["adapter_config"]["revision"] == config["trainer"]["model_revision"]


@pytest.mark.parametrize(
    "lora_override,expected_fragment",
    [
        ({"modules_to_save": ["lm_head"]}, "lm_head.weight"),
        ({"target_modules": ["embed_tokens"]}, "lora_embedding_A"),
        ({"trainable_token_indices": [3, 4]}, "trainable_tokens_delta"),
    ],
)
def test_explicit_peft_state_is_preserved_without_unpinned_embedding_lookup(
    tmp_path,
    monkeypatch,
    lora_override,
    expected_fragment,
):
    from plato.algorithms.lora import Algorithm
    from plato.models.huggingface import Model
    from plato.trainers.huggingface import Trainer

    directory = create_tiny_qwen3(tmp_path / "model")
    config = reference_config(directory)
    config["parameters"]["lora"].update(lora_override)
    with configure_environment(config):
        model = Model.get()
        trainer = Trainer(model=model)
        algorithm = Algorithm(trainer)

        def forbidden(*args, **kwargs):
            pytest.fail("Adapter export must not reread unpinned base config.")

        monkeypatch.setattr(type(model.config), "from_pretrained", forbidden)
        payload = algorithm.extract_weights()
        assert any(expected_fragment in key for key in payload)
        trainer.save_model("explicit.safetensors", str(tmp_path))
        saved_logits = trainer.require_model()(
            input_ids=torch.tensor([[1, 3, 4]])
        ).logits
        trainer.load_model("explicit.safetensors", str(tmp_path))
        torch.testing.assert_close(
            trainer.require_model()(input_ids=torch.tensor([[1, 3, 4]])).logits,
            saved_logits,
        )


def test_adapter_checkpoint_rejects_different_pin_and_preserves_active_model(tmp_path):
    from plato.algorithms.lora import Algorithm
    from plato.config import Config
    from plato.models.huggingface import Model
    from plato.trainers.huggingface import Trainer
    from tests.test_utils.qwen3 import tensor_digest

    directory = create_tiny_qwen3(tmp_path / "model")
    with configure_environment(reference_config(directory)):
        trainer = Trainer(model=Model.get())
        algorithm = Algorithm(trainer)
        before = tensor_digest(algorithm.extract_weights())
        trainer.save_model("pinned.safetensors", str(tmp_path))
        Config().trainer.model_revision = "different-pin"
        with pytest.raises(ValueError, match="identity mismatch"):
            trainer.load_model("pinned.safetensors", str(tmp_path))
        assert tensor_digest(algorithm.extract_weights()) == before


def test_peft_cpu_payload_snapshot_and_adapter_epoch_snapshot_are_independent(tmp_path):
    from plato.algorithms.lora import Algorithm
    from plato.models.huggingface import Model
    from plato.trainers.huggingface import Trainer
    from tests.test_utils.qwen3 import frozen_digest, tensor_digest

    directory = create_tiny_qwen3(tmp_path / "model")
    with configure_environment(reference_config(directory)):
        trainer = Trainer(model=Model.get())
        algorithm = Algorithm(trainer)
        payload = algorithm.extract_weights()
        payload_before = tensor_digest(payload)
        base_before = frozen_digest(trainer.model)
        trainer.save_model("1_1_1.0.safetensors")
        with torch.no_grad():
            for parameter in trainer.require_model().parameters():
                if parameter.requires_grad:
                    parameter.add_(0.1)
        assert tensor_digest(payload) == payload_before
        active_before = tensor_digest(algorithm.extract_weights())
        snapshot = trainer.obtain_model_at_time(1, 2.0)
        assert tensor_digest(algorithm.extract_weights(snapshot)) == payload_before
        assert tensor_digest(algorithm.extract_weights()) == active_before
        assert frozen_digest(snapshot) == frozen_digest(trainer.model) == base_before


def test_tokenizer_expansion_retains_embedding_checkpoint_state(tmp_path):
    from transformers import AutoTokenizer

    from plato.algorithms.lora import Algorithm
    from plato.models.huggingface import Model
    from plato.trainers.huggingface import Trainer

    directory = create_tiny_qwen3(tmp_path / "model")
    tokenizer = AutoTokenizer.from_pretrained(directory)
    assert tokenizer is not None
    tokenizer.add_tokens([f"added-token-{index}" for index in range(128)])
    tokenizer.save_pretrained(tmp_path / "expanded-tokenizer")
    config = reference_config(directory)
    config["trainer"]["tokenizer_name"] = str(tmp_path / "expanded-tokenizer")
    with configure_environment(config):
        trainer = Trainer(model=Model.get())
        assert trainer.require_model().get_input_embeddings().num_embeddings == len(
            tokenizer
        )
        algorithm = Algorithm(trainer)
        payload = algorithm.extract_weights()
        assert any(key.endswith("embed_tokens.weight") for key in payload)
        trainer.save_model("expanded.safetensors", str(tmp_path))
        trainer.load_model("expanded.safetensors", str(tmp_path))
        restored = algorithm.extract_weights()
        for key in payload:
            torch.testing.assert_close(restored[key], payload[key])


def test_configured_seed_reproduces_actual_lora_initialization(tmp_path):
    from plato.algorithms.lora import Algorithm
    from plato.models.huggingface import Model
    from plato.trainers.huggingface import Trainer
    from tests.test_utils.qwen3 import tensor_digest

    directory = create_tiny_qwen3(tmp_path / "model")
    with configure_environment(reference_config(directory)):
        first = Algorithm(Trainer(model=Model.get())).extract_weights()
        torch.manual_seed(999)
        second = Algorithm(Trainer(model=Model.get())).extract_weights()
        assert tensor_digest(first) == tensor_digest(second)


def test_supplied_preexpanded_peft_model_preserves_embedding_snapshots_and_logits(
    tmp_path,
    monkeypatch,
):
    from transformers import AutoTokenizer

    from plato.algorithms.lora import Algorithm
    from plato.models.huggingface import Model
    from plato.serialization.safetensor import deserialize_tree
    from plato.trainers.huggingface import Trainer
    from tests.test_utils.qwen3 import tensor_digest

    directory = create_tiny_qwen3(tmp_path / "model")
    tokenizer = AutoTokenizer.from_pretrained(directory)
    assert tokenizer is not None
    tokenizer.add_tokens([f"added-token-{index}" for index in range(128)])
    tokenizer.save_pretrained(tmp_path / "expanded-tokenizer")
    config = reference_config(directory)
    config["trainer"]["tokenizer_name"] = str(tmp_path / "expanded-tokenizer")
    with configure_environment(config):
        model = Model.get()
        original_vocabulary = model.config.vocab_size
        assert len(tokenizer) > original_vocabulary
        # The caller already resized the model and changed model.config.vocab_size.
        # Trainer must compare with its independently pinned original AutoConfig.
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        trainer = Trainer(model=model)
        assert trainer.config.vocab_size == original_vocabulary
        assert model.config.vocab_size == len(tokenizer)
        algorithm = Algorithm(trainer)

        def forbidden(*args, **kwargs):
            pytest.fail("Embedding export must not reload an unpinned base config.")

        monkeypatch.setattr(type(model.config), "from_pretrained", forbidden)
        payload = algorithm.extract_weights()
        before = tensor_digest(payload)
        assert any(key.endswith("embed_tokens.weight") for key in payload)
        assert any(key.endswith("lm_head.weight") for key in payload)
        trainer.save_model("preexpanded.safetensors", str(tmp_path))
        saved = deserialize_tree((tmp_path / "preexpanded.safetensors").read_bytes())
        assert any(key.endswith("embed_tokens.weight") for key in saved)
        assert any(key.endswith("lm_head.weight") for key in saved)
        model.eval()
        inputs = torch.tensor([[1, len(tokenizer) - 1, 4]])
        with torch.no_grad():
            expected = model(input_ids=inputs).logits.detach().clone()
            model.get_input_embeddings().weight[-1].add_(0.5)
            model.get_output_embeddings().weight[-1].add_(0.3)
        assert tensor_digest(payload) == before
        with torch.no_grad():
            assert not torch.allclose(model(input_ids=inputs).logits, expected)
        algorithm.load_weights(payload)
        with torch.no_grad():
            torch.testing.assert_close(
                model(input_ids=inputs).logits, expected, rtol=0, atol=0
            )
            model.get_input_embeddings().weight[-1].add_(0.5)
            model.get_output_embeddings().weight[-1].add_(0.3)
        trainer.load_model("preexpanded.safetensors", str(tmp_path))
        with torch.no_grad():
            torch.testing.assert_close(
                model(input_ids=inputs).logits, expected, rtol=0, atol=0
            )
