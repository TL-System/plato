# Retired Older Llama2 and OPT split recipes

The Llama2-7B and OPT350m split configurations are historical recipes removed by the approved consolidation. Shared split-learning and attack code, all three GPT2 recipes, and ordinary OPT federated LoRA remain active. This does not assert that the model families or split learning are intrinsically obsolete.

Maintained replacement: [examples/split_learning/llm_split_learning/split_learning_wikitext2_gpt2.toml](../../../examples/split_learning/llm_split_learning/split_learning_wikitext2_gpt2.toml).

## Preserved source

- [examples/split_learning/llm_split_learning/split_learning_wikitext2_llama2.toml](original/examples/split_learning/llm_split_learning/split_learning_wikitext2_llama2.toml)
- [examples/split_learning/llm_split_learning/split_learning_wikitext2_opt350m.toml](original/examples/split_learning/llm_split_learning/split_learning_wikitext2_opt350m.toml)

The [manifest](manifest.json) records every original path, Git blob and mode, SHA256, byte count and moved-versus-copied disposition. Source commit: c1359992f70533c0f2a47d30742f9b14295da882; tree: 9e75b3b2f306677fabc6f9214658bfeacd152dc5. Copied runtime files and the pre-change qualification ledger are historical context.

The [license](original/LICENSE), [package manifest](original/pyproject.toml) and [lockfile](original/uv.lock) preserve the historical environment, without establishing compatibility or successful execution.

## Historical restoration

Create a separate checkout:

```bash
git worktree add --detach ../plato-history-older-split-llm-recipes c1359992f70533c0f2a47d30742f9b14295da882
```

Use original paths in that checkout; do not import archived modules into the maintained runtime. External data/checkpoints must be supplied separately. Original documentation and source retain their wording, bibliography and historical commands; those commands have not been requalified.

See the [archive index](../README.md).
