set -eu
unset PLATO_DOCS_ENVIRONMENT PLATO_DOCS_BOOTSTRAP_ENVIRONMENT UV_PROJECT_ENVIRONMENT
test ! -e .venv-docs
test ! -e .venv-docs-bootstrap
test ! -e .venv
test ! -e docs/site
test ! -e ci-artifacts
uv run --no-project --python 3.13 python .github/scripts/check_distribution.py > /tmp/plato-p2-docs-environment-fix/release-package.log 2>&1
test ! -e .venv
cat /tmp/plato-p2-docs-environment-fix/release-package.log
