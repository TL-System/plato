To install Material for MkDocs, run:

```bash
uv venv
source .venv/bin/activate
uv pip install mkdocs-material mike
```

To set the latest version when there is a version upgrade, use:

```bash
mike deploy --push --update-aliases 1.4.2 latest
mike set-default 1.4.2
```

To serve the website for development, run:

```bash
mike serve
```

To compile it to a static website, run:

```bash
mkdocs build
```

The static website will be available in `/sites` directory.
