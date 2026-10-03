# Evaluators

Plato's evaluator subsystem adds **structured benchmark outputs** on top of the trainer's normal scalar test metric.

A trainer's `TestingStrategy` still returns a single value such as accuracy or perplexity. When `[evaluation]` is configured, Plato then runs an evaluator and stores richer benchmark results in the trainer context for server-side logging.

## Runtime flow

The evaluation path is:

1. `TestingStrategy.test_model(...)` computes the trainer's scalar test metric.
2. `plato.evaluators.runner.run_configured_evaluation(...)` reads `Config().evaluation`.
3. The evaluator registry resolves the configured factory by name, unless a
   matching evaluator instance was supplied through `evaluator_override`.
4. The evaluator returns an `EvaluationResult`.
5. Plato stores the serialized payload in `TrainingContext.state` under:
   - `evaluation_results`
   - `evaluation_primary`
6. Server logging flattens those metrics into CSV columns prefixed with `evaluation_`.

This means evaluator metrics participate in the normal results CSV without requiring a separate sidecar log.

## Core abstractions

### `EvaluationInput`

Defined in `plato/evaluators/base.py`.

This object carries the data an evaluator may need:

- `model`
- `trainer`
- `tokenizer`
- `context`
- `config`
- `testset`
- `sampler`
- `local_metric`

Not every evaluator needs every field. For example, Lighteval mainly needs the model, tokenizer, context, and evaluator config.

### `EvaluationResult`

Also defined in `plato/evaluators/base.py`.

An evaluator returns:

- `evaluator`: short name such as `lighteval`
- `primary_metric`: metric name to highlight
- `metrics`: normalized numeric summary metrics
- `higher_is_better`: per-metric comparison hints
- `metadata`: optional structured detail payloads
- `artifacts`: optional external artifact references

Plato validates that `primary_metric` exists inside `metrics` and adds a derived `primary_value` when serializing the result.

### `Evaluator`

Custom evaluators subclass `plato.evaluators.base.Evaluator` and implement:

```python
def evaluate(self, request: EvaluationInput) -> EvaluationResult:
    ...
```

## Built-in evaluators

| Name | Class | Registration | Notes |
| --- | --- | --- | --- |
| `lighteval` | `plato.evaluators.lighteval.LightevalEvaluator` | Auto-registered via `registry.register` | Server-side LLM evaluation through Hugging Face Lighteval. |

## Evaluator registry

`plato.evaluators.registry` provides the registration surface:

- `register(name, factory)`
- `unregister(name)`
- `registered_names()`
- `get(config=None, allow_missing=False)`

Factories receive the evaluator config node and return an evaluator instance.

Example:

```python
from plato.evaluators import registry
from plato.evaluators.base import EvaluationInput, EvaluationResult, Evaluator


class TokenCountEvaluator(Evaluator):
    def evaluate(self, request: EvaluationInput) -> EvaluationResult:
        token_count = float(request.context.state.get("token_count", 0.0))
        return EvaluationResult(
            evaluator="token_count",
            primary_metric="token_count",
            metrics={"token_count": token_count},
            higher_is_better={"token_count": False},
        )


registry.register("token_count", TokenCountEvaluator)
```

Then activate it from TOML:

```toml
[evaluation]
type = "token_count"
```

## Failure handling

`run_configured_evaluation(...)` treats evaluator failures as **non-fatal by default**.

- If `evaluation.fail_on_error = false` or omitted, Plato logs the exception and continues without structured evaluation metrics.
- If `evaluation.fail_on_error = true`, the exception is raised and the run stops.

These settings govern exceptions raised while evaluating. Unknown or retired
evaluator selections fail during resolution so a misspelled or unsupported
backend cannot silently produce an unevaluated run.

## Lighteval-specific behaviour

Plato's Lighteval adapter adds several integration details on top of upstream Lighteval:

- task aliases are mapped to the concrete upstream ids:
  - `arc_easy` → `arc:easy`
  - `arc_challenge` → `arc:challenge`
  - `piqa` → custom `piqa_hf`
- summary metrics are normalized into stable Plato names:
  - `ifeval_avg`
  - `hellaswag`
  - `arc_easy`
  - `arc_challenge`
  - `arc_avg`
  - `piqa`
- detailed per-task metrics are also exported to the CSV, for example:
  - `evaluation_ifeval_prompt_level_strict_acc`
  - `evaluation_arc_easy_acc`
  - `evaluation_arc_challenge_acc_stderr`
- safe runtime defaults are used for server-side evaluation:
  - `batch_size = 1`
  - `model_parallel = false`
  - `device = Config.device()`
  - `dtype` inferred from `trainer.bf16` / `trainer.fp16`
- when `trainer.max_concurrency` spawns a subprocess for testing, Plato persists evaluator state so the parent server process still logs the structured metrics.

The adapter also exposes the preset `smollm_round_fast`, used by the SmolLM2 server-side evaluation example.

Task scoring follows upstream Lighteval metric definitions. Plato maps those
outputs to stable summary names and detailed CSV columns; normalization does
not change the upstream scoring rules.

Current-model exports and configured local fallbacks receive fresh response
caches per evaluation. Local fallback directories are independent full copies,
with cleanup on success and failure; source artifacts remain untouched. See
[model artifacts and response caches](../configurations/evaluation.md#model-artifacts-and-response-caches)
for storage costs and concurrent-writer limits.

## Optional runtime qualification

Provision the locked extra, test dependencies, and the exact required NLTK
resources before running the complete optional suite:

```bash
uv sync --locked --python 3.13 --extra llm_eval --group test
uv run --locked --extra llm_eval python -m nltk.downloader punkt punkt_tab
uv run --locked --extra llm_eval --group test python -m pytest tests/llm_eval --test-profile=llm-eval -ra
```

The strict profile requires its complete reviewed case inventory and successful
execution, with no skips, xfails, or xpasses. Missing dependencies or resources
fail during preflight; preflight does not download models or resources.
`--collect-only` checks prerequisites and inventory, not runtime behavior.
Filters such as `-k`, `-m`, and collection exclusions are incompatible with
complete qualification. Use an explicit filesystem path without the profile
for a focused run:

```bash
uv run --locked --extra llm_eval --group test python -m pytest tests/llm_eval/test_lighteval_runtime.py -k model_config -ra
```

Focused runs retain prerequisite and strict outcome checks but do not qualify
the full inventory. Native MLX and Lighteval optional tests run separately;
optional `--pyargs` selection and cross-profile targets are unsupported.

For combined core and Lighteval qualification, the retained model-search tests
also need the `test-model-search` group, which includes `test` and `ptflops`:

```bash
uv run --locked --extra llm_eval --group test-model-search python -m pytest tests --test-profile=llm-eval -ra
```

Ordinary core, `base`, `mandatory`, and `mlx-native` requests exclude `tests/llm_eval` before importing its modules.
Existing unit tests in `tests/evaluators` remain core tests and do not establish
actual Lighteval runtime qualification.

The real runtime regression fixtures use a tiny local GPT-2 model, local parquet
tasks, and controlled one-token generation on CPU with Python 3.13. They exercise
the actual registry, runner, Pipeline, metrics, cache freshness, and cleanup.
Their scores demonstrate regression behavior, not public benchmark accuracy,
performance, downloaded Hub-model quality, GPU, or distributed qualification.

## Server logging contract

Evaluator metrics appear in the runtime CSV under the `evaluation_` prefix.

- Every key in `EvaluationResult.metrics` becomes `evaluation_<metric>`.
- Lighteval additionally exports detailed task-level metrics from the evaluator metadata.
- The CSV schema expands automatically when new evaluation columns appear.

See [Evaluation](../configurations/evaluation.md) for configuration details and [Results](../configurations/results.md) for logging behaviour.

## Extending Plato with new evaluators

A good custom evaluator should:

1. Accept a lightweight config object.
2. Use `request.context.state` for temporary coordination instead of mutating the trainer directly.
3. Return a normalized `EvaluationResult` with small summary metrics in `metrics`.
4. Store heavier nested details inside `metadata` if they are still useful for downstream inspection.
5. Choose a stable `primary_metric` so CSV dashboards and automated comparisons have a clear headline number.

For larger integrations, pair the evaluator with a dedicated documentation example under `docs/docs/examples/` and a smoke test under `tests/evaluators/`.
