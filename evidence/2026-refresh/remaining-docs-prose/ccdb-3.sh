uv --version
PLATO_CLUSTER_PYTHON="$(command -v python)"
UV_PYTHON_DOWNLOADS=never uv sync --locked --python "$PLATO_CLUSTER_PYTHON"
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
