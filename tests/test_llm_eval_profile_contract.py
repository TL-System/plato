"""Keep optional evaluation isolated and require exact qualification evidence."""

import ast
import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from tests.test_mlx_profile_contract import policy, profile_source, project, run_pytest


@dataclass(frozen=True)
class _ChildResult:
    status: int
    output: str
    receipt: dict


def _production_child(tmp_path, arguments, *, directory="") -> _ChildResult:
    root = Path(__file__).resolve().parents[1]
    receipt = tmp_path / "imports.json"
    script = """
import importlib.abc, importlib.metadata, json, sys
from pathlib import Path
import pytest
events = []
class Audit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lighteval" or fullname.startswith("lighteval."):
            events.append(fullname)
            raise RuntimeError("real Lighteval import sentinel")
        return None
sys.meta_path.insert(0, Audit())
code = int(pytest.main(sys.argv[2:] +
    ["--collect-only", "-q", "-p", "no:cacheprovider"]))
try:
    version = importlib.metadata.version("lighteval")
except importlib.metadata.PackageNotFoundError:
    version = None
real_modules = {name: getattr(module, "__file__", None)
                for name, module in sys.modules.items()
                if name == "lighteval" or name.startswith("lighteval.")}
mandatory_packages = {}
for name in ("opacus", "kazoo", "gymnasium"):
    try:
        mandatory_packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        mandatory_packages[name] = None
Path(sys.argv[1]).write_text(json.dumps({"events": events, "version": version,
    "modules": real_modules, "production_conftest": "tests.conftest" in sys.modules,
    "runtime_modules": [n for n in sys.modules if "test_lighteval_runtime" in n],
    "mandatory_packages": mandatory_packages,
    "status": code}))
raise SystemExit(code)
"""
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "tests")])
    result = subprocess.run(
        [sys.executable, "-c", script, str(receipt), *arguments],
        cwd=root / directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert receipt.exists(), output
    return _ChildResult(result.returncode, output, json.loads(receipt.read_text()))


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["tests"],
        ["tests", "--test-profile=base", "-m", "not runtime"],
        ["tests", "--test-profile=mandatory", "-m", "runtime"],
        [
            "tests",
            "--test-profile=mandatory",
            "-m",
            "runtime and not retained_model_search",
        ],
        ["--pyargs", "tests.test_config_loader"],
    ],
)
def test_real_production_core_graph_never_imports_optional(tmp_path, arguments):
    result = _production_child(tmp_path, arguments)
    requires_mandatory = (
        "--test-profile=base" not in arguments and "--pyargs" not in arguments
    )
    missing = next(
        (
            name
            for name, version in result.receipt["mandatory_packages"].items()
            if version is None
        ),
        None,
    )
    if requires_mandatory and missing is not None:
        assert result.status == pytest.ExitCode.USAGE_ERROR, result.output
        assert (
            f"mandatory prerequisite failed: No module named '{missing}'"
            in result.output
        )
    else:
        assert result.status == 0, result.output
    assert result.receipt["production_conftest"]
    assert result.receipt["events"] == []
    # Core fake-dependency tests can leave in-memory stand-ins. Their origins
    # must never resolve to an installed Lighteval file.
    assert all(origin is None for origin in result.receipt["modules"].values())
    assert result.receipt["runtime_modules"] == []


@pytest.mark.parametrize(
    "selector,directory",
    [
        ("tests.llm_eval.test_lighteval_runtime", ""),
        ("llm_eval.test_lighteval_runtime", "tests"),
        ("test_lighteval_runtime", "tests/llm_eval"),
    ],
)
def test_real_pyargs_alias_backstop_precedes_optional_import(
    tmp_path, selector, directory
):
    result = _production_child(tmp_path, ["--pyargs", selector], directory=directory)
    assert result.status != 0
    assert "llm-eval selector unsupported" in result.output
    assert result.receipt["events"] == []
    assert result.receipt["runtime_modules"] == []


@pytest.fixture(scope="module")
def optional_ci():
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/pytorch_qualification.yml"
    ).read_text()
    step = workflow.split("- name: Lighteval CPU qualification", 1)[1].split(
        "- name: Ruff and fatal configured typing", 1
    )[0]
    snippets = step.split("python - <<'PY'\n")[1:]
    source = textwrap.dedent(snippets[2].split("\n          PY", 1)[0])
    tree = ast.parse(source)
    assert isinstance(tree.body[-1], ast.Raise)
    namespace = {"__name__": "isolated_optional_ci"}
    exec(
        compile(
            ast.Module(body=tree.body[:-1], type_ignores=[]),
            "optional-ci-verifier",
            "exec",
        ),
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "missing",
        "duplicate",
        "extra",
        "skip",
        "failure",
        "error",
        "xfail",
        "xpass",
        "identity",
        "source",
        "junit-duplicate",
    ],
)
def test_optional_ci_gate_rejects_false_success(
    optional_ci, tmp_path, monkeypatch, damage
):
    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setenv("NLTK_DATA", str(tmp_path / "nltk"))
    identity = {"fixture": "identity"}
    monkeypatch.setitem(
        optional_ci.verify_llm_eval.__globals__, "checked_identity", lambda: identity
    )
    ledger = root / "tests/llm_eval/cases.json"
    nodes = [case["nodeid"] for case in json.loads(ledger.read_text())["cases"]]
    collected = list(nodes)
    if damage == "missing":
        collected.pop()
    elif damage == "duplicate":
        collected.append(nodes[0])
    elif damage == "extra":
        collected.append(nodes[0] + "_extra")
    (tmp_path / "llm-eval-collection.log").write_text("\n".join(collected))
    suites = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(
        suites,
        "testsuite",
        tests=str(len(nodes)),
        failures="0",
        errors="0",
        skipped="0",
    )
    for index, node in enumerate(
        nodes + ([nodes[0]] if damage == "junit-duplicate" else [])
    ):
        module, name = node.split("::", 1)
        case = ElementTree.SubElement(
            suite, "testcase", name=name, classname=module[:-3].replace("/", ".")
        )
        if index == 0 and damage in {"skip", "failure", "error", "xfail"}:
            ElementTree.SubElement(
                case, "skipped" if damage in {"skip", "xfail"} else damage
            )
    ElementTree.ElementTree(suites).write(tmp_path / "llm-eval.xml")
    (tmp_path / "llm-eval-ty.log").write_text(
        json.dumps(identity) + "\nAll checks passed!\n"
    )
    (tmp_path / "llm-eval.log").write_text(
        json.dumps(identity)
        + "\n"
        + ("10 passed, 1 xpassed" if damage == "xpass" else "11 passed")
        + "\nLighteval complete qualification\n"
    )
    import hashlib

    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    source = {
        "interpreter": {} if damage == "identity" else identity,
        "sha256": {
            "tests/llm_eval/cases.json": "bad" if damage == "source" else digest
        },
    }
    (tmp_path / "llm-eval-source-ledger.json").write_text(json.dumps(source))
    (tmp_path / "llm-eval-prerequisites.json").write_text(
        json.dumps(
            {
                "interpreter": identity,
                "preflight_passed": True,
                "resources": {
                    name: {"resolved": str(tmp_path / "nltk" / name)}
                    for name in ("punkt", "punkt_tab")
                },
            }
        )
    )
    assert optional_ci.verify_llm_eval(tmp_path) == (0 if damage == "none" else 1)
    accepted = json.loads((tmp_path / "llm-eval-acceptance.json").read_text())
    assert accepted["accepted"] is (damage == "none")


@pytest.mark.parametrize("mode", ["correct", "omitted", "wrong"])
def test_actual_uv_selects_persistent_environment_before_import(
    optional_ci, tmp_path, mode
):
    # uv --no-sync must choose the task environment despite a distinguishable
    # default .venv. No installation or project sync occurs in this regression.
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(tmp_path / ".venv")],
        check=True,
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="identity-fixture"\nversion="0.0.0"\n'
        'requires-python=">=3.13"\n'
    )
    # The same exact workflow identity function is executed in the chosen child.
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/pytorch_qualification.yml"
    ).read_text()
    step = workflow.split("- name: Lighteval CPU qualification", 1)[1]
    source = textwrap.dedent(
        step.split("python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    )
    source = source.split(
        'Path("ci-artifacts/llm-eval-interpreter.json").write_text', 1
    )[0]
    environment = os.environ.copy()
    if mode == "omitted":
        environment.pop("UV_PROJECT_ENVIRONMENT", None)
    else:
        environment["UV_PROJECT_ENVIRONMENT"] = (
            sys.prefix if mode == "correct" else str(tmp_path / ".venv")
        )
    # The negative wrong mode uses the correct interpreter with a wrong configured
    # environment, while omitted mode exercises actual uv default selection.
    command = (
        [sys.executable]
        if mode == "wrong"
        else ["uv", "run", "--no-sync", "--python", "3.13", "python"]
    )
    result = subprocess.run(
        command + ["-c", source + "\nprint(interpreter_identity())"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert (result.returncode == 0) is (mode == "correct"), (
        result.stdout + result.stderr
    )


@pytest.mark.parametrize("omitted", ["punkt", "punkt_tab"])
def test_real_resource_absence_fails_before_registry(
    tmp_path, omitted, record_property
):
    root = Path(__file__).parents[1]
    script = """
import hashlib, importlib.abc, importlib.metadata, importlib.util, json, shutil, sys
from pathlib import Path
import pytest
root, fixture, omitted = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location(
    "actual_profile", root / "tests/conftest.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    importlib.metadata.version("lighteval")
except importlib.metadata.PackageNotFoundError:
    try:
        module._eval_prerequisites()
    except pytest.UsageError as exc:
        assert "llm-eval prerequisite failed" in str(exc)
        print(json.dumps({"actual_optional_absence": True, "error": str(exc)}))
        raise SystemExit(0)
    raise AssertionError("Actual absent optional package unexpectedly passed")
import nltk
other = "punkt_tab" if str(omitted) == "punkt" else "punkt"
origin = Path(str(nltk.data.find("tokenizers/" + other)))
destination = fixture / "tokenizers" / other
shutil.copytree(origin, destination)
nltk.data.path[:] = [str(fixture)]
nltk.data.clear_cache()
assert nltk.data.path == [str(fixture)]
lookup_error = None
try:
    nltk.data.find("tokenizers/" + str(omitted))
except LookupError as exc:
    lookup_error = str(exc)
assert lookup_error is not None, "Negative fixture accidentally finds missing resource"
actual_other = str(nltk.data.find("tokenizers/" + other))
events = []
class NoRegistry(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lighteval" or fullname.startswith("lighteval."):
            events.append(fullname)
            raise AssertionError("Registry imported before missing resource rejection")
        return None
sys.meta_path.insert(0, NoRegistry())
def forbidden_download(*args, **kwargs):
    events.append("download")
    raise AssertionError("Unexpected resource download")
nltk.download = forbidden_download
try:
    module._eval_prerequisites()
except pytest.UsageError as exc:
    assert "missing NLTK resource tokenizers/" + str(omitted) in str(exc), str(exc)
    assert events == [], events
    print(json.dumps({"executable": sys.executable, "prefix": sys.prefix,
        "nltk_version": importlib.metadata.version("nltk"),
        "nltk_data_sha256": hashlib.sha256(
            Path(nltk.data.__file__).read_bytes()).hexdigest(),
        "effective_paths": nltk.data.path, "actual_lookup_error": lookup_error,
        "other_lookup": actual_other, "events": events, "error": str(exc),
        "profile_sha256": hashlib.sha256(
            (root / "tests/conftest.py").read_bytes()).hexdigest(),
        "fixture_sha256": {str(p.relative_to(fixture)):
                           hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(fixture.rglob("*")) if p.is_file()}}))
else:
    raise AssertionError("Missing actual resource unexpectedly passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), str(tmp_path / "resource"), omitted],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record_property("actual_resource_guard", result.stdout)


def _config(root, args, profile=None, **options):
    values = {
        "test_profile": profile,
        "keyword": "",
        "markexpr": "",
        "deselect": [],
        "ignore": [],
        "ignore_glob": [],
        "pyargs": False,
        "collectonly": True,
        **options,
    }
    return SimpleNamespace(
        rootpath=root,
        invocation_params=SimpleNamespace(dir=root),
        args=args,
        getoption=values.get,
    )


@pytest.mark.parametrize("profile", [None, "base", "mandatory", "mlx-native"])
def test_unrequested_boundary_is_ignored_before_import(policy, tmp_path, profile):
    checks = policy._ProfileChecks(_config(tmp_path, ["tests"], profile))
    assert checks.pytest_ignore_collect(tmp_path / "tests/llm_eval/test_real.py")
    assert (
        checks.pytest_ignore_collect(tmp_path / "tests/test_llm_eval_suffix.py") is None
    )


@pytest.mark.parametrize("profile", ["base", "mandatory", "mlx-native"])
def test_eval_target_conflicts_before_preflight(policy, tmp_path, profile):
    with pytest.raises(pytest.UsageError, match="conflict"):
        policy._ProfileChecks(_config(tmp_path, ["tests/llm_eval"], profile))


@pytest.mark.parametrize(
    "args,profile",
    [
        (["tests/mlx_native"], "llm-eval"),
        (["tests/mlx_native", "tests/llm_eval"], None),
    ],
)
def test_cross_optional_families_fail_early(policy, tmp_path, args, profile):
    with pytest.raises(pytest.UsageError, match="conflict"):
        policy._ProfileChecks(_config(tmp_path, args, profile))


@pytest.mark.parametrize(
    "option,value",
    [
        ("keyword", "preset"),
        ("markexpr", "llm_eval"),
        ("markexpr", "runtime"),
        ("deselect", ["test_case"]),
        ("ignore", ["tests/test_other.py"]),
        ("ignore_glob", ["*failure*"]),
        ("pyargs", True),
    ],
)
def test_named_profile_rejects_every_filter(policy, tmp_path, option, value):
    with pytest.raises(pytest.UsageError):
        policy._ProfileChecks(
            _config(tmp_path, ["tests/llm_eval"], "llm-eval", **{option: value})
        )


@pytest.mark.parametrize("selector", ["tests.llm_eval", "tests.llm_eval.test_real"])
def test_canonical_pyargs_rejected_without_resolution(policy, tmp_path, selector):
    with pytest.raises(pytest.UsageError, match="selector unsupported"):
        policy._ProfileChecks(_config(tmp_path, [selector], pyargs=True))


def test_marker_alone_does_not_request_optional_imports(policy, tmp_path):
    with pytest.raises(pytest.UsageError, match="explicit"):
        policy._ProfileChecks(_config(tmp_path, ["tests"], markexpr="llm_eval"))


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate"])
def test_named_inventory_counter_cannot_qualify_partial_collection(
    policy, tmp_path, damage
):
    checks = policy._ProfileChecks(_config(tmp_path, ["tests/llm_eval"], "llm-eval"))
    node = "tests/llm_eval/test_real.py::test_real"
    checks.eval_expected = Counter([node])
    nodes = (
        []
        if damage == "missing"
        else [node, node if damage == "duplicate" else node + "_extra"]
    )
    session = SimpleNamespace(
        items=[
            SimpleNamespace(nodeid=n, path=tmp_path / n.split("::")[0]) for n in nodes
        ],
        exitstatus=pytest.ExitCode.OK,
    )
    checks.pytest_collection_finish(session)
    checks.pytest_sessionfinish(session, pytest.ExitCode.OK)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("llm-eval ledger" in issue for issue in checks.violations)


@pytest.mark.parametrize("outcome", ["skip", "xfail", "xpass"])
def test_focused_optional_outcomes_are_strict(policy, tmp_path, outcome):
    checks = policy._ProfileChecks(_config(tmp_path, ["tests/llm_eval/test_real.py"]))
    report = SimpleNamespace(
        nodeid="tests/llm_eval/test_real.py::test_real",
        when="call",
        skipped=outcome == "skip",
        passed=outcome == "xpass",
        longrepr=("file", 1, "Skipped: intentional"),
    )
    if outcome != "skip":
        report.wasxfail = "intentional"
    checks.pytest_runtest_logreport(report)
    assert checks.violations


@pytest.mark.parametrize("damage", ["empty", "duplicate", "traversal", "criteria"])
def test_ledger_rejects_invalid_inventory(policy, tmp_path, damage):
    case = {
        "nodeid": "tests/llm_eval/test_real.py::test_real",
        "criteria": ["LE1", "LE2", "LE3", "LE4"],
    }
    cases = [case]
    if damage == "empty":
        cases = []
    elif damage == "duplicate":
        cases.append(case)
    elif damage == "traversal":
        case["nodeid"] = "tests/llm_eval/../test_real.py::test_real"
    else:
        case["criteria"] = ["LE1"]
    ledger = tmp_path / "cases.json"
    ledger.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    with pytest.raises(pytest.UsageError, match="ledger invalid"):
        policy._load_eval_ledger(ledger)


def test_frozen_runtime_ledger_matches_reviewed_inventory(policy):
    root = Path(__file__).parents[1]
    inventory = json.loads(
        (
            root / "evidence/2026-refresh/lighteval-frozen-runtime-inventory.json"
        ).read_text()
    )
    assert policy._load_eval_ledger(root / "tests/llm_eval/cases.json") == Counter(
        inventory["runtime_nodes"]
    )


def test_optional_initializer_remains_inert():
    tree = ast.parse((Path(__file__).parent / "llm_eval/__init__.py").read_text())
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[0].value.value, str)


@pytest.fixture
def eval_project(project):
    boundary = project.tests / "llm_eval"
    boundary.mkdir()
    (boundary / "__init__.py").write_text('"""Inert optional package."""\n')
    (boundary / "test_real.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', [1, 2], ids=['one', 'two'])\n"
        "def test_real(value):\n    assert value > 0\n"
    )
    cases = [
        {
            "nodeid": f"tests/llm_eval/test_real.py::test_real[{name}]",
            "criteria": ["LE1", "LE2", "LE3", "LE4"],
        }
        for name in ("one", "two")
    ]
    (boundary / "cases.json").write_text(
        json.dumps({"schema_version": 1, "cases": cases})
    )
    with (project.tests / "conftest.py").open("a") as stream:
        stream.write("\ndef _eval_prerequisites():\n    pass\n")
    with (project.root / "pytest.ini").open("a") as stream:
        stream.write("    llm_eval: optional profile\n")
    return project


@pytest.mark.parametrize(
    "arguments",
    [
        ["tests/llm_eval", "--test-profile=llm-eval"],
        ["tests", "--test-profile=llm-eval"],
        ["tests/llm_eval", "--test-profile=llm-eval", "--collect-only"],
        ["tests/llm_eval/test_real.py", "-k", "one"],
    ],
)
def test_named_and_focused_hooks_use_distinct_qualification(eval_project, arguments):
    result = run_pytest(eval_project, *arguments)
    assert result.returncode == 0, result.output
    if "--collect-only" in arguments:
        assert (
            "collection-only validation (no execution qualification)" in result.output
        )
    elif "-k" in arguments:
        assert "focused execution (not complete qualification)" in result.output
    else:
        assert "Lighteval complete qualification" in result.output


@pytest.mark.parametrize(
    "damage", ["subset", "dropped", "duplicate", "core-only", "environment-filter"]
)
def test_real_hooks_reject_incomplete_named_qualification(eval_project, damage):
    target = "tests/llm_eval"
    if damage == "subset":
        target += "/test_real.py::test_real[one]"
    elif damage == "core-only":
        target = "tests/test_core.py"
    elif damage == "dropped":
        with (eval_project.tests / "conftest.py").open("a") as stream:
            stream.write(
                "\ndef pytest_collection_modifyitems(items):\n    items.pop()\n"
            )
    elif damage == "duplicate":
        with (eval_project.tests / "conftest.py").open("a") as stream:
            stream.write(
                "\ndef pytest_collection_modifyitems(items):\n"
                "    items.append(items[0])\n"
            )
    result = run_pytest(
        eval_project,
        target,
        "--test-profile=llm-eval",
        addopts="-k one" if damage == "environment-filter" else None,
    )
    assert result.returncode != 0
    assert "Lighteval complete qualification" not in result.output


@pytest.mark.parametrize(
    "statement",
    [
        "pytest.skip('collect', allow_module_level=True)",
        "pytest.importorskip('deliberately_absent_llm_eval')",
        "@pytest.mark.skip(reason='setup')\ndef test_real():\n    pass",
        "@pytest.mark.xfail(reason='non-strict')\ndef test_real():\n    assert False",
        "@pytest.mark.xfail(reason='xpass')\ndef test_real():\n    pass",
        "@pytest.mark.xfail(reason='strict', strict=True)\ndef test_real():\n    pass",
        "def test_real():\n    pytest.skip('call')",
        "@pytest.fixture(autouse=True)\ndef broken():\n"
        "    raise RuntimeError('setup')\ndef test_real():\n    pass",
        "@pytest.fixture(autouse=True)\ndef broken():\n    yield\n"
        "    raise RuntimeError('teardown')\ndef test_real():\n    pass",
    ],
)
def test_focused_real_report_hooks_reject_every_nonpassing_outcome(
    eval_project, statement
):
    target = eval_project.tests / "llm_eval/test_real.py"
    target.write_text("import pytest\n" + statement + "\n")
    result = run_pytest(eval_project, "tests/llm_eval/test_real.py")
    assert result.returncode != 0, result.output
    assert "Lighteval complete qualification" not in result.output


def test_stager_omits_optional_weights_and_keeps_all_required_records(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "optional_stager",
        Path(__file__).parents[1] / ".github/scripts/stage_ci_artifacts.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    for name in module.EXPECTED_RECORDS:
        (source / name).write_text("evidence")
    fixture = source / "llm-eval-tmp"
    fixture.mkdir()
    (fixture / "weights.safetensors").write_bytes(b"weights")
    (fixture / "diagnostic.json").write_text("{}")
    (source / "pytest-current").symlink_to(fixture, target_is_directory=True)
    record = module.stage_artifacts(source, tmp_path / "staged")
    included = [item["path"] for item in record["included"]]
    assert all(included.count(name) == 1 for name in module.EXPECTED_RECORDS)
    assert "llm-eval-tmp/diagnostic.json" in included
    assert "llm-eval-tmp/weights.safetensors" not in included
    assert not any("pytest-current" in name for name in included)
