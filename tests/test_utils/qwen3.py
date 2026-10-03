"""Actual HF/PEFT Qwen3 runtime checks shared by offline and pinned qualification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import get_peft_model_state_dict
from transformers import TrainerCallback

from plato.algorithms.lora import Algorithm
from plato.config import Config
from plato.datasources.lora import DataSource
from plato.models.huggingface import Model
from plato.trainers.huggingface import Trainer

REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = REPOSITORY / "configs/HuggingFace/fedavg_qwen3_06b_lora.toml"


def reference_config(model_directory: Path | None = None) -> dict:
    """Read the supported reference, making fixture paths independent of cwd."""
    with REFERENCE_CONFIG.open("rb") as handle:
        config = tomllib.load(handle)
    for split, filename in config["data"]["data_files"].items():
        config["data"]["data_files"][split] = str(REPOSITORY / filename)
    if model_directory is not None:
        for key in ("model_name", "tokenizer_name"):
            config["trainer"][key] = str(model_directory)
    return config


def create_tiny_qwen3(directory: Path) -> Path:
    """Save random native Qwen3 weights and a real local tokenizer, offline."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    torch.manual_seed(17)
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    texts = []
    for split in ("train", "validation"):
        texts.extend(
            item["text"]
            for item in json.loads(
                (REPOSITORY / "tests/fixtures/qwen3" / f"{split}.json").read_text()
            )
        )
    tokenizer.train_from_iterator(
        texts,
        trainer=WordLevelTrainer(
            special_tokens=["<pad>", "<eos>", "<unk>"],
            vocab_size=192,
        ),
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=128,
        model_input_names=["input_ids", "attention_mask"],
    )
    fast_tokenizer.save_pretrained(directory)
    # Deliberately preserve a padded embedding vocabulary larger than tokenizer.
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            attention_dropout=0.0,
            pad_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(directory)
    return directory


def tensor_digest(state) -> str:
    """Hash tensor names, metadata and bytes without copying the whole base."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        array = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def frozen_digest(model) -> str:
    return tensor_digest(
        {
            name: value
            for name, value in model.named_parameters()
            if not value.requires_grad
        }
    )


class GradientProbe(TrainerCallback):
    """Observe actual PEFT gradients immediately before the optimizer update."""

    def __init__(self):
        self.updates = 0
        self.finite_gradients = True
        self.nonzero_gradients = False

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        assert model is not None
        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert grads, "No adapter gradients reached the optimizer."
        self.finite_gradients &= all(torch.isfinite(g).all().item() for g in grads)
        self.nonzero_gradients |= any(torch.count_nonzero(g).item() for g in grads)
        self.updates += 1


def logits(trainer, dataset):
    batch, _ = trainer._collate_wrapper([dataset[0]])
    trainer.model.eval()
    with torch.no_grad():
        return trainer.model(**batch, return_dict=True).logits.detach().clone()


def validate_runtime(checkpoint_directory: Path) -> dict:
    """Train serial 2/3 clients, run real server aggregation and reload adapters."""
    from plato.servers.fedavg import Server

    torch.manual_seed(17)
    torch.set_num_threads(4)
    started = time.perf_counter()
    datasource = DataSource()
    model = Model.get()
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ), "A base parameter is unexpectedly trainable."
    original_embedding_size = model.get_input_embeddings().num_embeddings
    base_before = frozen_digest(model)
    probe = GradientProbe()
    trainer = Trainer(model=model, callbacks=[probe])
    assert model.get_input_embeddings().num_embeddings == original_embedding_size
    assert frozen_digest(model) == base_before, "Trainer changed pretrained base."
    algorithm = Algorithm(trainer)
    initial = algorithm.extract_weights()
    assert initial and all("lora_" in key for key in initial)
    live = get_peft_model_state_dict(
        algorithm._peft_base(model), save_embedding_layers=False
    )
    for key, tensor in initial.items():
        assert tensor.device.type == "cpu" and not tensor.requires_grad
        assert tensor.data_ptr() != live[key].data_ptr(), "Mutable CPU payload alias."
    trainset, testset = datasource.get_train_set(), datasource.get_test_set()
    assert len(trainset) == 5 and len(testset) == 2
    batch, labels = trainer._collate_wrapper([trainset[0]])
    assert labels is not None and (labels[batch["attention_mask"] == 0] == -100).all()
    assert torch.equal(
        labels[batch["attention_mask"] == 1],
        batch["input_ids"][batch["attention_mask"] == 1],
    )
    config = Config().trainer._asdict()
    before_perplexity = trainer.test_model(config, testset)
    assert math.isfinite(before_perplexity)
    updates = []
    immutable_first = None
    for client_id, indices in ((1, [0, 1]), (2, [2, 3, 4])):
        algorithm.load_weights(initial)
        assert tensor_digest(algorithm.extract_weights()) == tensor_digest(initial)
        trainer.set_client_id(client_id)
        client_start = time.perf_counter()
        trainer.train_model(config, trainset, indices)
        payload = algorithm.extract_weights()
        if client_id == 1:
            immutable_first = {key: value.clone() for key, value in payload.items()}
        else:
            assert immutable_first is not None
            for key in payload:
                torch.testing.assert_close(
                    updates[0].payload[key], immutable_first[key], rtol=0, atol=0
                )
        assert tensor_digest(payload) != tensor_digest(initial)
        assert frozen_digest(model) == base_before
        train_loss = trainer.run_history.get_latest_metric("train_loss")
        perplexity = trainer.test_model(config, testset)
        assert math.isfinite(train_loss) and math.isfinite(perplexity)
        updates.append(
            SimpleNamespace(
                client_id=client_id,
                payload=payload,
                report=SimpleNamespace(
                    num_samples=len(indices),
                    accuracy=perplexity,
                    processing_time=0.0,
                    comm_time=0.0,
                    training_time=time.perf_counter() - client_start,
                    train_loss=train_loss,
                ),
            )
        )
    assert probe.finite_gradients and probe.nonzero_gradients and probe.updates == 5
    algorithm.load_weights(initial)
    # Exercise the production server's full report processing and aggregation.
    # Global evaluation is then called directly to keep this qualification in one
    # bounded process; socket/process orchestration has separate integration gates.
    Config().server.do_test = False
    server = Server()
    server.model = model
    server.trainer = trainer
    server.algorithm = algorithm
    server.context.algorithm = algorithm
    server.updates = updates
    server.current_round = 1
    asyncio.run(server._process_reports())
    global_payload = algorithm.extract_weights()
    differences = []
    for key, tensor in global_payload.items():
        expected = (2 * updates[0].payload[key] + 3 * updates[1].payload[key]) / 5
        torch.testing.assert_close(tensor, expected, rtol=1e-6, atol=1e-8)
        unweighted = (updates[0].payload[key] + updates[1].payload[key]) / 2
        differences.append((tensor - unweighted).abs().max().item())
    assert max(differences) > 1e-8, "Updates did not distinguish weighted averaging."
    global_perplexity = trainer.test_model(config, testset)
    assert math.isfinite(global_perplexity)
    assert frozen_digest(model) == base_before
    expected_logits = logits(trainer, testset)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    trainer.save_model("global.safetensors", str(checkpoint_directory))
    from plato.serialization.safetensor import deserialize_tree

    saved = deserialize_tree((checkpoint_directory / "global.safetensors").read_bytes())
    assert saved and all("lora_" in key for key in saved)
    algorithm.load_weights(initial)
    trainer.load_model("global.safetensors", str(checkpoint_directory))
    torch.testing.assert_close(
        logits(trainer, testset), expected_logits, rtol=1e-6, atol=1e-6
    )
    assert frozen_digest(model) == base_before
    assert tensor_digest(algorithm.extract_weights()) == tensor_digest(global_payload)
    metadata = json.loads((checkpoint_directory / "global.safetensors.hf").read_text())
    assert metadata["artifacts"]["model_revision"] == Config().trainer.model_revision
    return {
        "model_class": type(model.get_base_model()).__name__,
        "configured_artifacts": metadata["artifacts"],
        "device": str(trainer.device),
        "dtype": str(next(model.parameters()).dtype),
        "seed": 17,
        "max_length": Config().data.max_length,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_updates": probe.updates,
        "client_sample_counts": [update.report.num_samples for update in updates],
        "client_train_losses": [update.report.train_loss for update in updates],
        "before_perplexity": before_perplexity,
        "global_perplexity": global_perplexity,
        "before_adapter_sha256": tensor_digest(initial),
        "after_adapter_sha256": tensor_digest(global_payload),
        "frozen_base_sha256": base_before,
        "frozen_base_unchanged": True,
        "finite_nonzero_gradients": True,
        "payloads_independent": True,
        "adapter_only_checkpoint": True,
        "reload_logits_match": True,
        "weighted_oracle_all_tensors": True,
        "weighted_vs_unweighted_max_difference": max(differences),
        "elapsed_seconds": time.perf_counter() - started,
    }
