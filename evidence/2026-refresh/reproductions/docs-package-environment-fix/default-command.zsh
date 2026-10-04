set -eu
unset PLATO_DOCS_ENVIRONMENT PLATO_DOCS_BOOTSTRAP_ENVIRONMENT UV_PROJECT_ENVIRONMENT
test ! -e .venv-docs
test ! -e .venv-docs-bootstrap
test ! -e .venv
mkdir -p ci-artifacts/docs
git rev-parse HEAD > ci-artifacts/docs/commit.txt
shasum -a 256 pyproject.toml uv.lock docs/build.sh docs/requirements.txt > ci-artifacts/docs/source-hashes.txt
PLATO_DOCS_PYTHON="$(uv python find 3.13)" PLATO_DOCS_ARTIFACT_DIR=ci-artifacts/docs ./docs/build.sh > /tmp/plato-p2-docs-environment-fix/default-docs.log 2>&1
test -x .venv-docs/bin/python
test ! -e .venv-docs-bootstrap
test ! -e .venv
printf '%s\n' '{"fixture":"default-docs-environment-retained"}' > .venv-docs/p2-m3-environment-sentinel.json
printf '%s\n' '{"fixture":"default-generated-site-retained"}' > docs/site/p2-m3-site-sentinel.json
mkdir -p ci-artifacts/older
printf '%s\n' '{"fixture":"default-ci-output-retained"}' > ci-artifacts/older/p2-m3-artifact-sentinel.json
/tmp/plato-p2-docs-environment-fix/controller-env/bin/python /tmp/plato-p2-docs-environment-fix/environment-snapshot.py before default
uv run --no-project --python 3.13 python .github/scripts/check_distribution.py > /tmp/plato-p2-docs-environment-fix/default-package.log 2>&1
/tmp/plato-p2-docs-environment-fix/controller-env/bin/python /tmp/plato-p2-docs-environment-fix/environment-snapshot.py after default
cat /tmp/plato-p2-docs-environment-fix/default-package.log
