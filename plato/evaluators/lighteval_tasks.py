"""Custom Lighteval task definitions used by Plato."""

from __future__ import annotations

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def piqa_hf_prompt(line, task_name: str | None = None):
    """Prompt formatter for a parquet-backed PIQA mirror on the Hugging Face Hub."""
    letters = list(ascii_uppercase)[:2]
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['goal']}\n"
    query += "".join(
        [f"{key}. {choice}\n" for key, choice in zip(letters, [line["sol1"], line["sol2"]])]
    )
    query += "Answer: "

    return Doc(
        task_name=task_name,
        query=query,
        choices=letters,
        gold_index=int(line["label"]),
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


piqa_hf = LightevalTaskConfig(
    name="piqa_hf",
    prompt_function=piqa_hf_prompt,
    hf_repo="gimmaru/piqa",
    hf_subset="default",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [piqa_hf]
