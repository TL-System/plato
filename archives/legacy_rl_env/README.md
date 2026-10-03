# Legacy Gym environment

`rl_env.py` is the unchanged historical module formerly located at
`plato/utils/rl_env.py`. It was archived during the 2026 refresh after a tracked
source/configuration scan found no active imports or configured entrypoints.
The active `plato/utils/reinforcement_learning/` package and FEI example remain
separately maintained; no active runtime imports this archive.

Source: [TL-System/Plato](https://github.com/TL-System/plato), exact pre-move
baseline `fda60becedfa6236c11d415b87a868b52789a41e`. The last source modification
was `da8869512e417002975a16f818bcad7e91224279`. Original upstream references and
comments are retained in the file. The repository's Apache-2.0 license is copied
to `LICENSE` here.

SHA-256 identities:

- `rl_env.py`: `557fec876520d089a69c18484bcc73465eb1f52ac332442acc35427d57c7a40c`
- `LICENSE`: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`

This placeholder returns a four-value Gym step result, wraps an already shaped
state in an extra dimension, drives an event loop synchronously, and waits
forever once the episode limit is reached. These historical limitations were
preserved, not repaired. It is not a supported Gymnasium integration.

For historical inspection, create a separate checkout of the baseline above;
the original path and its tracked `uv.lock` are available there. That lock keeps
Python 3.13 as the default and includes Gymnasium through the RL extra/dev group.
There was no active run command or configuration for this adapter. A historical
checkout preserves its context but does not establish compatibility or fix the
limitations above.
