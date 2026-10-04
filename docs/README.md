# Building and previewing the documentation

Run commands from the repository root with Python 3.13 and uv available.
The [installation guide](docs/install.md#building-the-documentation) describes
the locked docs environment and its configurable paths.

Build the static website:

```bash
PLATO_DOCS_PYTHON="$(uv python find 3.13)" ./docs/build.sh
```

The build script provisions `.venv-docs` from the root lockfile, checks dependency
parity, and runs MkDocs in strict mode. The HTML output is in `docs/site`.
It provisions `.venv-docs-bootstrap` only if uv 0.12.22 is not already available.

After a successful build, preview edits with the provisioned docs interpreter:

```bash
.venv-docs/bin/python -m mkdocs serve --strict -f docs/mkdocs.yml
```

Open `http://127.0.0.1:8000/` and stop the server with `Ctrl-C` when finished.
If you set `PLATO_DOCS_ENVIRONMENT` for the build, use that environment's
`bin/python` in the serve command as well.
