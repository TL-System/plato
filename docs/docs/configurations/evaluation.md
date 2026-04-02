# Evaluation

Plato supports an optional `[evaluation]` section for **structured server-side evaluation**. This runs **after** the trainer's regular test metric (for example accuracy or perplexity) and records named benchmark metrics under the `evaluation_` prefix in the runtime CSV.

Use this section when you want benchmark-style outputs such as IFEval, ARC, HellaSwag, PIQA, or Nanochat CORE instead of only a single scalar test metric.

## When evaluation runs

Structured evaluation is triggered from the trainer's test flow, so it depends on server-side testing being enabled:

```toml
[server]
do_test = true
```

If `[evaluation]` is omitted, Plato only records the trainer's normal scalar metric.

## Common options

!!! example "type"
    The evaluator backend to run.

    Built-in values include:

    - `lighteval` for Hugging Face's Lighteval benchmark runner.
    - `nanochat_core` for Nanochat's CORE benchmark. **Requires `trainer.type = "nanochat"`.**
      This evaluator is not registered in the general evaluator registry; it is wired
      internally by the nanochat trainer. Using it with any other trainer type produces
      no evaluation output and no error.

!!! example "fail_on_error"
    Whether evaluator failures should abort the run.

    Default value: `false`

    When `false`, Plato logs the evaluator exception and continues without structured evaluation metrics. Set this to `true` when the evaluation itself is a required part of the experiment.

## Built-in evaluators

| Evaluator | Install path | Primary output style | Typical use |
| --- | --- | --- | --- |
| `lighteval` | `uv sync --extra llm_eval` | Named benchmark metrics such as `ifeval_avg` and `arc_avg` | Server-side LLM evaluation |
| `nanochat_core` | `uv sync --extra nanochat` | `core_metric` | Nanochat benchmark runs — requires `trainer.type = "nanochat"` and a trained Nanochat tokenizer under `~/.cache/nanochat/tokenizer/` |

## Lighteval

Plato's Lighteval adapter wraps the `lighteval` package and normalizes its task outputs into CSV-friendly metrics.

### Supported options

!!! example "preset"
    Name of the built-in task preset.

    Current built-in value:

    - `smollm_round_fast`

    This preset runs:

    - `ifeval`
    - `hellaswag`
    - `arc_easy`
    - `arc_challenge`
    - `piqa`

!!! example "primary_metric"
    The summary metric to treat as the evaluator's primary output.

    For `smollm_round_fast`, the default is `ifeval_avg`.

!!! example "backend"
    Lighteval execution backend.

    Supported values in Plato's current integration include:

    - `transformers`
    - `accelerate`

    `transformers` and `accelerate` currently resolve to the same safe server-side launcher path in Plato.

!!! example "batch_size"
    Evaluation batch size passed to Lighteval.

    Default value: `1`

    Plato intentionally defaults to `1` to avoid aggressive auto-probing on multi-GPU systems.

!!! example "max_length"
    Optional maximum sequence length passed to the Lighteval transformers backend.

!!! example "max_samples"
    Optional **per-task** sample cap.

    Example: `max_samples = 32` runs up to 32 examples for each configured task. Lighteval shuffles deterministically before truncating, so the subset is stable across runs.

    !!! warning "Partial benchmark"
        When `max_samples` is set, benchmark numbers are partial and should not be compared directly with full-dataset leaderboard runs.

!!! example "model_parallel"
    Whether Lighteval should shard the evaluated model across multiple GPUs.

    Default value: `false`

!!! example "dtype"
    Optional evaluation dtype override.

    If omitted, Plato infers a sensible default from the trainer configuration:

    - `trainer.bf16 = true` → `bfloat16`
    - `trainer.fp16 = true` → `float16`

!!! example "device"
    Device string for evaluation, such as `cuda:0`, `cuda:1`, or `cpu`.

    If omitted, Plato uses `Config.device()`.

!!! example "show_progress"
    Whether to show the coarse-grained server-side Lighteval progress bar.

    Default value: `true`

### Reference example

The configuration `configs/HuggingFace/fedavg_smol_smoltalk_smollm2_135m.toml` uses Lighteval like this:

```toml
[server]
do_test = true

[evaluation]
type = "lighteval"
preset = "smollm_round_fast"
primary_metric = "ifeval_avg"
backend = "transformers"
batch_size = 1
model_parallel = false
device = "cuda:0"
show_progress = true
max_samples = 32
```

### Metrics exported to the CSV

Lighteval summary metrics are written as:

- `evaluation_ifeval_avg`
- `evaluation_hellaswag`
- `evaluation_arc_easy`
- `evaluation_arc_challenge`
- `evaluation_arc_avg`
- `evaluation_piqa`

Plato also exports detailed Lighteval task metrics as additional CSV columns when they are present, for example:

- `evaluation_ifeval_prompt_level_strict_acc`
- `evaluation_ifeval_inst_level_loose_acc`
- `evaluation_arc_easy_acc`
- `evaluation_arc_challenge_acc_stderr`
- `evaluation_hellaswag_em`
- `evaluation_piqa_em`

These columns are added to the CSV automatically the first time they appear.

## Nanochat CORE

Nanochat's CORE benchmark is also available through `[evaluation]`.

!!! note "Tokenizer required"
    `nanochat_core` does not just need the `nanochat` Python dependencies. It also
    requires a trained Nanochat tokenizer under `~/.cache/nanochat/tokenizer/`
    (notably `tokenizer.pkl` and `token_bytes.pt`).

    Plato can download the CORE evaluation bundle automatically, but it does **not**
    create the tokenizer automatically.

    See [Nanochat in Plato](examples/case-studies/5. Nanochat in Plato.md) for the
    full setup sequence, including tokenizer training.

### Supported options

!!! example "bundle_dir"
    Optional directory containing the downloaded CORE evaluation bundle.

    If omitted, Plato resolves the Nanochat base directory automatically and downloads the bundle when needed.

!!! example "max_per_task"
    Optional cap on the number of examples per CORE task.

    Default value: `-1`, which means use all available examples.

### Example

!!! warning "Requires the nanochat trainer"
    `nanochat_core` is only wired up when `trainer.type = "nanochat"`. The nanochat
    trainer creates the evaluator internally rather than looking it up in the registry.
    Setting `[evaluation] type = "nanochat_core"` with any other trainer type silently
    produces no evaluation output.

```toml
[trainer]
type = "nanochat"

[evaluation]
type = "nanochat_core"
max_per_task = 16
```

This evaluator exports `core_metric`, which can be listed in `[results].types`.

## Results logging

Structured evaluator metrics are written directly into the runtime CSV in `result_path`. The CSV is the authoritative log for evaluator outputs.

See [Results](results.md) for how evaluator columns are named and expanded at runtime.
