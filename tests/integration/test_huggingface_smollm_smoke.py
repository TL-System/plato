from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict

from plato.config import Config
from tests.integration.utils import async_run


class SmokeChatTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"
    model_max_length = 128
    role_ids = {"system": 71, "user": 72, "assistant": 73}

    def __len__(self):
        return 16

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False
    ):
        if not tokenize:
            return "".join(
                f"<{message['role']}>{message['content']}|" for message in messages
            )

        tokens = []
        for message in messages:
            tokens.append(self.role_ids[message["role"]])
            tokens.extend(100 + ord(char) for char in message["content"])
            tokens.append(0)
        if add_generation_prompt:
            tokens.append(99)
        return tokens

    def pad(self, features, padding=True, return_tensors=None):
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": []}
        for feature in features:
            pad_width = max_len - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * pad_width
            )
            batch["attention_mask"].append(
                feature.get("attention_mask", [1] * len(feature["input_ids"]))
                + [0] * pad_width
            )
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }


class SmokeHFModel(nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 4)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        del attention_mask, kwargs
        batch, seq = input_ids.shape
        vocab_size = self.embedding.num_embeddings
        logits = torch.zeros((batch, seq, vocab_size), dtype=torch.float32)
        loss = torch.tensor(0.25, dtype=torch.float32)
        if labels is not None:
            loss = loss + labels.float().mean() * 0
        return SimpleNamespace(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, new_size: int):
        self.embedding = nn.Embedding(new_size, 4)
        return self.embedding

    def gradient_checkpointing_enable(self):
        return None


@pytest.mark.integration
def test_smollm_smoltalk_config_smoke(monkeypatch, tmp_path):
    """Smoke test the SmolLM2 + smol-smoltalk config with mocked HF/Lighteval hooks."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs/HuggingFace/fedavg_smol_smoltalk_smollm2_135m.toml"
    assert config_path.exists()

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"},
                        ]
                    ]
                }
            ),
            "test": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "v"},
                            {"role": "assistant", "content": "b"},
                        ]
                    ]
                }
            ),
        }
    )

    previous_env = os.environ.get("config_file")
    previous_argv = sys.argv[:]
    Config._instance = None
    os.environ["config_file"] = str(config_path)
    sys.argv = [sys.argv[0], "--config", str(config_path), "--base", str(tmp_path)]

    try:
        config = Config()
        Config.args.id = 0
        if "max_concurrency" in config.trainer:
            del config.trainer["max_concurrency"]

        from importlib import import_module

        processor_registry = import_module("plato.processors.registry")
        csv_processor = import_module("plato.utils.csv_processor")
        datasources_hf = import_module("plato.datasources.huggingface")
        trainers_hf = import_module("plato.trainers.huggingface")
        models_registry = import_module("plato.models.registry")
        lighteval_eval = import_module("plato.evaluators.lighteval")
        server_mod = import_module("plato.servers.fedavg")

        monkeypatch.setattr(
            processor_registry,
            "get",
            lambda *args, **kwargs: (None, None),
        )
        monkeypatch.setattr(
            csv_processor,
            "initialize_csv",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            datasources_hf,
            "load_dataset",
            lambda *args, **kwargs: dataset,
        )
        monkeypatch.setattr(
            datasources_hf,
            "load_from_disk",
            lambda *args, **kwargs: dataset,
        )
        monkeypatch.setattr(datasources_hf.os.path, "exists", lambda *args: False)
        monkeypatch.setattr(
            datasources_hf.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: SimpleNamespace(),
        )
        monkeypatch.setattr(
            datasources_hf.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: SmokeChatTokenizer(),
        )
        monkeypatch.setattr(
            trainers_hf.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: SimpleNamespace(),
        )
        monkeypatch.setattr(
            trainers_hf.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: SmokeChatTokenizer(),
        )
        monkeypatch.setattr(
            models_registry,
            "get",
            lambda *args, **kwargs: SmokeHFModel(),
        )
        monkeypatch.setattr(
            lighteval_eval,
            "_resolve_model_reference",
            lambda request, export_dir=None: lighteval_eval.LightevalModelReference(
                model_name="/tmp/mock-smollm",
                tokenizer_name="/tmp/mock-smollm",
            ),
        )
        monkeypatch.setattr(
            lighteval_eval,
            "_run_lighteval_pipeline",
            lambda **kwargs: {
                "ifeval": 0.31,
                "hellaswag": 0.44,
                "arc_easy": 0.35,
                "arc_challenge": 0.25,
                "piqa": 0.61,
            },
        )

        server = server_mod.Server()
        server.configure()

        trainer = server.trainer
        assert trainer is not None
        model = trainer.model
        assert model is not None
        weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        update = SimpleNamespace(
            client_id=1,
            report=SimpleNamespace(
                num_samples=1,
                accuracy=0.5,
                processing_time=0.1,
                comm_time=0.1,
                training_time=0.1,
                train_loss=0.25,
            ),
            payload=weights,
        )
        server.updates = [update]
        server.current_round = 0
        server.context.current_round = 0

        async_run(server._process_reports())
        logged = server.get_logged_items()

        assert config.evaluation.type == "lighteval"
        assert logged["train_loss"] == pytest.approx(0.25)
        assert logged["evaluation_ifeval_avg"] == pytest.approx(0.31)
        assert logged["evaluation_hellaswag"] == pytest.approx(0.44)
        assert logged["evaluation_arc_avg"] == pytest.approx(0.30)
        assert logged["evaluation_piqa"] == pytest.approx(0.61)
    finally:
        if previous_env is None:
            os.environ.pop("config_file", None)
        else:
            os.environ["config_file"] = previous_env
        sys.argv = previous_argv
        Config._instance = None
