set -eu
unset PLATO_DOCS_ENVIRONMENT PLATO_DOCS_BOOTSTRAP_ENVIRONMENT UV_PROJECT_ENVIRONMENT
test ! -e .venv-docs
test ! -e .venv-docs-bootstrap
test ! -e .venv
docs_python="$(uv python find 3.13)"
mkdir -p ci-artifacts/docs
git rev-parse HEAD > ci-artifacts/docs/commit.txt
shasum -a 256 pyproject.toml uv.lock docs/build.sh docs/requirements.txt > ci-artifacts/docs/source-hashes.txt
(
    export PATH=/usr/bin:/bin:/usr/sbin:/sbin
    if command -v uv >/dev/null 2>&1; then exit 99; fi
    PLATO_DOCS_PYTHON="$docs_python" PLATO_DOCS_ARTIFACT_DIR=ci-artifacts/docs ./docs/build.sh
) > /tmp/plato-p2-docs-environment-fix/bootstrap-docs.log 2>&1
test -x .venv-docs/bin/python
test -x .venv-docs-bootstrap/bin/uv
test ! -e .venv
printf '%s\n' '{"fixture":"bootstrap-docs-environment-retained"}' > .venv-docs/p2-m3-environment-sentinel.json
printf '%s\n' '{"fixture":"bootstrap-uv-environment-retained"}' > .venv-docs-bootstrap/p2-m3-bootstrap-sentinel.json
printf '%s\n' '{"fixture":"bootstrap-generated-site-retained"}' > docs/site/p2-m3-site-sentinel.json
mkdir -p ci-artifacts/older
printf '%s\n' '{"fixture":"bootstrap-ci-output-retained"}' > ci-artifacts/older/p2-m3-artifact-sentinel.json
/tmp/plato-p2-docs-environment-fix/controller-env/bin/python /tmp/plato-p2-docs-environment-fix/environment-snapshot.py before bootstrap
uv run --no-project --python 3.13 python .github/scripts/check_distribution.py > /tmp/plato-p2-docs-environment-fix/bootstrap-package.log 2>&1
/tmp/plato-p2-docs-environment-fix/controller-env/bin/python /tmp/plato-p2-docs-environment-fix/environment-snapshot.py after bootstrap
cat /tmp/plato-p2-docs-environment-fix/bootstrap-package.log
