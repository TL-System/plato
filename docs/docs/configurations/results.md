!!! example "types"
    The set of columns that will be written into the runtime `.csv` file.

    Common built-in values include:

    - `round`
    - `accuracy`
    - `accuracy_std`
    - `elapsed_time`
    - `comm_time`
    - `processing_time`
    - `round_time`
    - `comm_overhead`
    - `train_loss`
    - `core_metric`
    - `local_epoch_num`
    - `edge_agg_num`
    - `evaluation_primary_value`

    !!! note "Note"
        Use commas to separate them. The default is `round, accuracy, elapsed_time`.

    !!! note "Structured evaluators"
        When `[evaluation]` is configured, Plato automatically appends any new `evaluation_*` columns that appear at runtime. You do **not** need to predeclare every Lighteval task metric in `results.types`, although predeclaring the summary columns can keep the CSV order stable.

!!! example "result_path"
    The path to the result `.csv` files.

    Default value: `<base_path>/results/`, where `<base_path>` is specified in the `general` section.

!!! example "record_clients_accuracy"
    Whether a separate `*_accuracy.csv` file should be written for client-side test metrics.

    Default value: `false`

    This setting only has an effect when `clients.do_test = true`.

## Structured evaluation columns

Structured evaluators write their metrics directly into the runtime CSV using the `evaluation_` prefix.

Examples:

- `evaluation_ifeval_avg`
- `evaluation_hellaswag`
- `evaluation_arc_easy`
- `evaluation_arc_challenge`
- `evaluation_arc_avg`
- `evaluation_piqa`
- `evaluation_ifeval_prompt_level_strict_acc`
- `evaluation_arc_easy_acc`
- `evaluation_arc_challenge_acc_stderr`

Summary metrics come from `EvaluationResult.metrics`. Plato also exports `evaluation_primary_value`, which mirrors the evaluator's configured primary metric value. Some evaluators, such as Lighteval, also expose detailed task-level metrics that Plato flattens into additional CSV columns automatically.

### Example

```toml
[results]
types = "round, elapsed_time, accuracy, train_loss, evaluation_ifeval_avg, evaluation_hellaswag, evaluation_arc_avg, evaluation_piqa"
```

This predeclares the summary columns, and Plato adds any extra detailed `evaluation_*` columns later if the evaluator emits them.

## Per-client accuracy logging

If you want a separate `*_accuracy.csv` file containing one row per selected client, enable both:

```toml
[clients]
do_test = true

[results]
record_clients_accuracy = true
```

Client-side testing alone is not enough; the extra CSV is only written when `results.record_clients_accuracy = true` is also set.

## Logging format

The runtime CSV is the authoritative structured-evaluation log. Plato no longer writes a separate JSONL sidecar for evaluator outputs.
