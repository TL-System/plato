"""Bounded offline assets using the installed evaluation libraries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any


def make_model_and_tokenizer():
    """Create a tiny causal model whose preferred token is controllable."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocabulary = {"A": 0, "B": 1, "C": 2, "D": 3, "[EOS]": 4, "[UNK]": 5}
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="[EOS]",
        pad_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=64,
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=64,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=4,
            eos_token_id=4,
            pad_token_id=4,
        )
    )
    prefer_token(model, 0)
    return model, tokenizer


def prefer_token(model, token_id: int) -> None:
    """Set every output logit to zero except the chosen token's positive logit."""
    import torch

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.transformer.ln_f.bias.fill_(1)
        model.transformer.wte.weight[token_id].fill_(1)


def make_offline_tasks(root: Path) -> list[tuple[Any, Any]]:
    """Copy preset task configs with local data and one-token fixture generation.

    The upstream prompt functions and metric implementations remain intact. The
    constructed rows measure integration behavior, not benchmark accuracy.
    """
    from datasets import Dataset
    from lighteval.tasks.tasks.arc import arc_challenge, arc_easy
    from lighteval.tasks.tasks.hellaswag import hellaswag
    from lighteval.tasks.tasks.ifeval.main import ifeval

    from plato.evaluators.lighteval_tasks import piqa_hf

    labels = [0, 0, 1]
    task_rows = [
        (
            ifeval,
            [
                {
                    "prompt": f"Reply with {['A', 'B'][label]} for case {index}",
                    "instruction_id_list": ["keywords:existence"],
                    "kwargs": [{"keywords": [["A", "B"][label]]}],
                }
                for index, label in enumerate(labels)
            ],
        ),
        (
            hellaswag,
            [
                {
                    "activity_label": f"case {index}",
                    "ctx_a": "Choose",
                    "ctx_b": "one",
                    "endings": ["first", "second", "third", "fourth"],
                    "label": str(label),
                }
                for index, label in enumerate(labels)
            ],
        ),
        (
            piqa_hf,
            [
                {
                    "goal": f"case {index}",
                    "sol1": "first",
                    "sol2": "second",
                    "label": label,
                }
                for index, label in enumerate(labels)
            ],
        ),
    ]
    for task in (arc_easy, arc_challenge):
        task_rows.append(
            (
                task,
                [
                    {
                        "question": f"Choose for case {index}",
                        "choices": {"text": ["A", "B"], "label": ["A", "B"]},
                        "answerKey": ["A", "B"][label],
                    }
                    for index, label in enumerate(labels)
                ],
            )
        )

    tasks = []
    for task, rows in task_rows:
        directory = root / task.name.replace(":", "_")
        directory.mkdir(parents=True)
        Dataset.from_list(rows).to_parquet(str(directory / "validation.parquet"))
        tasks.append(
            (
                task,
                replace(
                    task,
                    hf_repo=str(directory),
                    hf_subset="default",
                    hf_revision=None,
                    hf_avail_splits=["validation"],
                    evaluation_splits=["validation"],
                    few_shots_split=None,
                    few_shots_select=None,
                    generation_size=1,
                ),
            )
        )
    return tasks
