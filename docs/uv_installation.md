# Working with Plato

Plato uses `uv` as its package manager, which is a modern, fast Python package manager that provides significant performance improvements over `conda` environments. To install `uv`, refer to its [official documentation](https://docs.astral.sh/uv/getting-started/installation/), or simply run the following commands:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

To upgrade `uv`, run the command:

```
uv self update
```

To start working with Plato, first clone its git repository:

```shell
git clone git@github.com:TL-System/plato.git
cd plato
```

Then install its dependencies:

```shell
uv sync
```

You may also need some additional packages:

```shell
uv sync --extra dev
```

This includes:
- `pytest` for testing
- `black` for code formatting
- `pylint` for code linting


### 4. GPU Support Configuration

**Check CUDA version (Linux/Windows):**
```shell
nvidia-smi
```

The PyTorch installation will automatically detect and use the appropriate CUDA version. uv handles PyTorch dependencies intelligently through the project configuration.

**For macOS (CPU-only):**
No additional configuration needed - PyTorch CPU version will be installed automatically.



## Migration from Conda

If you're migrating from a conda environment:

1. **Export your current environment** (for reference):
   ```shell
   conda env export > conda_environment.yml
   ```

2. **Follow this uv installation guide**

3. **Remove old conda environment** (optional):
   ```shell
   conda env remove -n plato
   ```


## Visual Studio Code Integration

If you use Visual Studio Code, install these recommended extensions:
- `Python` (Microsoft)
- `Black Formatter`
- `Pylint`

Add to your `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "workbench.editor.enablePreview": false
}
```
