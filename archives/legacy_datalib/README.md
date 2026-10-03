# Historical multimodal dataset helpers

This archive preserves all 14 files formerly under `plato/datasources/datalib/`,
byte for byte from Plato commit `fda60becedfa6236c11d415b87a868b52789a41e`.
It was retired during the 2026 refresh after checking imports, dynamic module
names, configurations, examples, tests and documentation: the only callers were
other files inside this bundle. No registered active datasource uses it. The
archive is outside the shipped `plato` package and has no active imports.

The path map is `plato/datasources/datalib/<relative-path>` to
`archives/legacy_datalib/<relative-path>`. Original code, comments and historical
import names are preserved. The Apache-2.0 repository license is copied here;
authors and exact change history remain available with
`git log --follow fda60bec -- plato/datasources/datalib/<relative-path>`. The
history includes contributions by Sijia Chen, Yuting Zhang and Baochun Li.
This archive does not relicense externally supplied datasets or tools.

For historical investigation, create a separate checkout at the full commit
above and use the original paths. Its tracked `uv.lock` and `.python-version`
record the baseline environment; they do not qualify this unused bundle for
current dependencies. Restore the original layout rather than importing this
archive from active runtime code. The bundle has no single maintained experiment
entrypoint, and its external video/audio/annotation datasets are not bundled.
Individual utilities require additional historical dependencies such as MMCV,
MMAction, pandas and scikit-image, plus FFmpeg and external dataset files. Some
utilities execute shell commands and use old multimedia APIs. Their current
runtime compatibility and complete end-to-end reproduction are unqualified.

Known defects are preserved, not repaired:

- REFER `getRefIds(image_ids=[1])` builds a nested list and fails with
  `TypeError: list indices must be integers or slices, not str` on a two-image
  local fixture.
- REFER `getAnnIds(ref_ids=[1])` discards its computed intersection: the same
  fixture returns `[11, 22]` rather than `[11]`.
- These are observed filtering defects. The historical typing diagnostics for
  `self.data` alone do not prove a runtime failure: it is a dictionary in the
  executed index/filter probe.

The probe outcomes, closure check and preservation verification are recorded in
`evidence/2026-refresh/phase2a-validation.json`. The shared archive index is owned
by the later archive/documentation phase.
