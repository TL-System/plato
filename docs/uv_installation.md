# Installation with UV

UV is a modern, fast Python package manager that provides significant performance improvements over conda and pip. This guide covers installation using UV as the package manager.

## Installation Steps

### 1. Install UV

**Linux/macOS (recommended):**
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternative installation via pip:**
```shell
pip install uv
```

After installation, restart your terminal or run:
```shell
source $HOME/.local/bin/env
```
For detailed installation instructions, refer to the [official UV documentation](https://docs.astral.sh/uv/getting-started/installation/).
### 2. Clone the Repository

```shell
git clone https://github.com/TL-System/plato.git
cd plato
```

### 3. Install Plato and Dependencies

**For regular use:**
```shell
uv sync
```

**For development with additional tools:**
```shell
uv sync --extra dev
```
This includes:
- `pytest` for testing
- `black` for code formatting
- `pylint` for code linting

**For TensorFlow support:**
```shell
uv sync --extra tensorflow
```

**For MindSpore support:**
```shell
uv sync --extra mindspore
```

### 4. GPU Support Configuration

**Check CUDA version (Linux/Windows):**
```shell
nvidia-smi
```

The PyTorch installation will automatically detect and use the appropriate CUDA version. UV handles PyTorch dependencies intelligently through the project configuration.

**For macOS (CPU-only):**
No additional configuration needed - PyTorch CPU version will be installed automatically.



## Migration from Conda

If you're migrating from a conda environment:

1. **Export your current environment** (for reference):
   ```shell
   conda env export > conda_environment.yml
   ```

2. **Follow this UV installation guide**

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


