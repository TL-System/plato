# Evaluation

Plato supports an optional `[evaluation]` section for **structured server-side evaluation**. This runs **after** the trainer's regular test metric (for example accuracy or perplexity) and records named benchmark metrics under the `evaluation_` prefix in the runtime CSV.

Use this section when you want benchmark-style outputs such as IFEval, ARC, HellaSwag, or PIQA instead of only a single scalar test metric.

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

!!! example "fail_on_error"
    Whether evaluator failures should abort the run.

    Default value: `false`

    When `false`, Plato logs the evaluator exception and continues without structured evaluation metrics. Set this to `true` when the evaluation itself is a required part of the experiment.

Unknown or retired evaluator types fail during resolution. `fail_on_error`
controls failures during evaluation, not unsupported backend selection.

## Built-in evaluators

| Evaluator | Install path | Primary output style | Typical use |
| --- | --- | --- | --- |
| `lighteval` | `uv sync --locked --python 3.13 --extra llm_eval` | Named benchmark metrics such as `ifeval_avg` and `arc_avg` | Server-side LLM evaluation |

## Lighteval

Plato's Lighteval adapter wraps the `lighteval` package and normalizes its task outputs into CSV-friendly metrics.

Install the `llm_eval` extra and the NLTK `punkt` and `punkt_tab` resources as
shown in [Installation](../install.md#optional-server-side-llm-evaluation-with-lighteval).
Include `--extra llm_eval` on syncing `uv run` commands. Use a separate
environment for the incompatible `ssl` extra.

### Model artifacts and response caches

When the current model and tokenizer both provide `save_pretrained()`, Plato
exports them to a fresh temporary directory for each evaluation. This ensures
that later federated rounds evaluate their current weights rather than reuse
responses from an earlier round.

Otherwise, the adapter falls back to `trainer.model_name` and
`trainer.tokenizer_name` (the latter defaults to the model reference). Existing
local directories are copied in full into independent temporary directories;
if both references resolve to the same directory, it is copied once. The
configured source directories, including read-only sources, remain unchanged.
Large checkpoints and existing cache files can make this fallback expensive
in disk space and copy time. The normal current-model export path does not
perform this additional copy.

Plato uses fresh per-evaluation response-cache configuration and removes its
owned temporary model copies, exports, output directories, and response caches
on success or failure. This does not clear ordinary Hugging Face download
caches. Keep local source artifacts stable during copying: the fallback does
not provide an atomic snapshot while another process writes checkpoints.

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

## Results logging

Structured evaluator metrics are written directly into the runtime CSV in `result_path`. The CSV is the authoritative log for evaluator outputs.

See [Results](results.md) for how evaluator columns are named and expanded at runtime.
