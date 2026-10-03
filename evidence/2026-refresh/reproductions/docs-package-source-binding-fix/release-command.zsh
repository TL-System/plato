set -eu
test ! -e docs/site
test ! -e ci-artifacts
/tmp/plato-p2-source-binding-fix/controller-env/bin/python .github/scripts/check_distribution.py > /tmp/plato-p2-source-binding-fix/release-default-checker.log 2>&1
cat /tmp/plato-p2-source-binding-fix/release-default-checker.log
