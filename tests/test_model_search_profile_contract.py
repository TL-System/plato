"""Bound the serialized core and retained model-search qualification policies."""

import ast
import importlib.util
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from tests.test_mlx_profile_contract import policy, profile_source

CORE_RUNTIME = "runtime and not retained_model_search"
FAST_CORE = "not runtime and not slow"
RETAINED_MODULE = "tests/integration/test_retained_model_search.py"
RETAINED_NAMES = (
    "test_retained_width_round[anycostfl]",
    "test_retained_width_round[fedrolex]",
    "test_retained_width_round[heterofl]",
    "test_retained_activated_budget[anycostfl]",
    "test_retained_activated_budget[fedrolex]",
    "test_retained_activated_budget[heterofl]",
    "test_sysheterofl_subnet_round",
    "test_dlg_model_update[lenet]",
    "test_dlg_model_update[resnet_18]",
    "test_local_retired_selection[anycostfl]",
    "test_local_retired_selection[fedrolex]",
    "test_local_retired_selection[heterofl]",
    "test_local_retired_selection[dlg]",
    "test_retained_entrypoint_import[anycostfl-root]",
    "test_retained_entrypoint_import[anycostfl-example]",
    "test_retained_entrypoint_import[fedrolex-root]",
    "test_retained_entrypoint_import[fedrolex-example]",
    "test_retained_entrypoint_import[heterofl-root]",
    "test_retained_entrypoint_import[heterofl-example]",
)


@dataclass(frozen=True)
class _Item:
    nodeid: str
    path: Path
    markers: frozenset[str] = frozenset()

    def get_closest_marker(self, name: str) -> str | None:
        return name if name in self.markers else None


def _config(root, partition, *, args=None, profile="mandatory", **options):
    values = {
        "test_profile": profile,
        "markexpr": partition,
        "keyword": "",
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
        args=["tests"] if args is None else args,
        getoption=values.get,
    )


def _item(root, nodeid, *markers):
    return _Item(nodeid, root / nodeid.split("::", 1)[0], frozenset(markers))


def _startup(policy, root):
    return [
        _item(
            root,
            policy._STARTUP + name + (f"[{index}]" if count > 1 else ""),
            "runtime",
        )
        for name, count in policy._STARTUP_CASES.items()
        for index in range(count)
    ]


def _retained(root):
    return [
        _item(root, f"{RETAINED_MODULE}::{name}", "runtime", "retained_model_search")
        for name in RETAINED_NAMES
    ]


def _finish(checks, items, *, passed=True):
    session = SimpleNamespace(items=items, exitstatus=pytest.ExitCode.OK)
    checks.pytest_collection_modifyitems(items)
    checks.pytest_collection_finish(session)
    if passed:
        checks.passed.update(item.nodeid for item in items)
    checks.pytest_sessionfinish(session, pytest.ExitCode.OK)
    return session


@pytest.mark.parametrize(
    "partition", ["", "runtime", "not runtime", CORE_RUNTIME, FAST_CORE]
)
@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize(
    "runtime,retained", [(False, False), (False, True), (True, False), (True, True)]
)
def test_exact_partition_deselection_truth_table(
    policy, tmp_path, partition, runtime, retained, slow
):
    checks = policy._ProfileChecks(_config(tmp_path, partition))
    markers = {
        name
        for name, enabled in [
            ("runtime", runtime), ("retained_model_search", retained), ("slow", slow)
        ]
        if enabled
    }
    checks.pytest_deselected(
        [_item(tmp_path, "tests/test_scope.py::test_case", *markers)]
    )
    permitted = {
        "": False,
        "runtime": not runtime,
        "not runtime": runtime,
        CORE_RUNTIME: not runtime or retained,
        FAST_CORE: runtime or slow,
    }[partition]
    assert bool(checks.violations) is not permitted


@pytest.mark.parametrize(
    "partition", ["", "runtime", "not runtime", CORE_RUNTIME, FAST_CORE]
)
def test_accepted_full_partitions_keep_required_inventories(
    policy, tmp_path, partition
):
    checks = policy._ProfileChecks(_config(tmp_path, partition))
    runtime = _startup(policy, tmp_path)
    required = [_item(tmp_path, node) for node in policy._REQUIRED]
    items = required if partition in {"not runtime", FAST_CORE} else runtime
    if partition == "":
        items += required
    session = _finish(checks, items)
    assert session.exitstatus == pytest.ExitCode.OK, checks.violations
    assert not checks.violations


@pytest.mark.parametrize(
    "partition",
    [
        "runtime and not retained_model_search and not startup_containment",
        "runtime and (not retained_model_search)",
        "not retained_model_search and runtime",
        "runtime or retained_model_search",
        "not slow",
        "not slow and not runtime",
        "not runtime and (not slow)",
        "not runtime and not slow and not integration",
    ],
)
def test_unapproved_full_partition_is_rejected_even_without_deselection(
    policy, tmp_path, partition
):
    with pytest.raises(pytest.UsageError, match="unapproved full-suite partition"):
        policy._ProfileChecks(_config(tmp_path, partition))


@pytest.mark.parametrize(
    "damage", ["runtime-node", "startup-count", "retained-startup", "not-passed"]
)
def test_core_runtime_required_presence_and_pass_checks(policy, tmp_path, damage):
    checks = policy._ProfileChecks(
        _config(tmp_path, CORE_RUNTIME, collectonly=damage != "not-passed")
    )
    items = _startup(policy, tmp_path)
    if damage == "runtime-node":
        items = [item for item in items if item.nodeid not in policy._RUNTIME]
    elif damage == "startup-count":
        items.append(_item(tmp_path, items[0].nodeid + "[extra]", "runtime"))
    elif damage == "retained-startup":
        removed = items.pop(0)
        checks.pytest_deselected(
            [_item(tmp_path, removed.nodeid, "runtime", "retained_model_search")]
        )
        assert not checks.violations
    session = _finish(checks, items, passed=damage != "not-passed")
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("required" in violation for violation in checks.violations)


@pytest.mark.parametrize(
    "profile,partition",
    [("base", "not runtime"), ("mandatory", "not runtime"), ("mandatory", FAST_CORE)],
)
def test_non_runtime_required_mpc_omission_still_fails(
    policy, tmp_path, profile, partition
):
    checks = policy._ProfileChecks(_config(tmp_path, partition, profile=profile))
    omitted = "tests/mpc/test_mpc.py::test_round_info_store_local"
    session = _finish(
        checks, [_item(tmp_path, node) for node in policy._REQUIRED if node != omitted]
    )
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any(omitted in violation for violation in checks.violations)


@pytest.mark.parametrize("collectonly", [False, True])
@pytest.mark.parametrize("partition", ["not runtime", FAST_CORE])
def test_mandatory_non_runtime_keeps_dp_presence_and_pass_checks(
    policy, tmp_path, collectonly, partition
):
    checks = policy._ProfileChecks(
        _config(tmp_path, partition, collectonly=collectonly)
    )
    nodeid = policy._DP_MODULE + "::test_dp_strategy_handles_plato_sampler_get"
    items = [_item(tmp_path, node) for node in policy._REQUIRED]
    if collectonly:
        items = [item for item in items if item.nodeid != nodeid]
    else:
        checks.passed.update(item.nodeid for item in items if item.nodeid != nodeid)
    session = _finish(checks, items, passed=collectonly)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any(nodeid in violation for violation in checks.violations)


@pytest.mark.parametrize("profile", [None, "base", "mlx-native", "llm-eval"])
def test_fast_partition_requires_mandatory_profile(policy, tmp_path, profile):
    with pytest.raises(pytest.UsageError, match="requires mandatory profile"):
        policy._ProfileChecks(_config(tmp_path, FAST_CORE, profile=profile))


def test_fast_partition_rejects_keyword_filter(policy, tmp_path):
    with pytest.raises(pytest.UsageError, match="does not permit keyword filters"):
        policy._ProfileChecks(_config(tmp_path, FAST_CORE, keyword="test_case"))


@pytest.mark.parametrize("option", ["ignore", "ignore_glob", "deselect"])
def test_fast_partition_rejects_collection_exclusions(policy, tmp_path, option):
    checks = policy._ProfileChecks(
        _config(tmp_path, FAST_CORE, **{option: ["tests/test_other.py"]})
    )
    session = _finish(checks, [_item(tmp_path, node) for node in policy._REQUIRED])
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert "test profiles do not permit collection exclusions" in checks.violations


@pytest.mark.parametrize("outcome", ["skip", "xfail", "xpass"])
def test_fast_partition_rejects_nonpassing_outcomes(policy, tmp_path, outcome):
    checks = policy._ProfileChecks(_config(tmp_path, FAST_CORE, collectonly=False))
    report = SimpleNamespace(
        nodeid="tests/test_scope.py::test_case",
        skipped=outcome != "xpass",
        passed=outcome == "xpass",
        when="call",
        longrepr=("file", 1, "Skipped: injected"),
    )
    if outcome != "skip":
        report.wasxfail = "injected"
    checks.pytest_runtest_logreport(report)
    session = _finish(checks, [_item(tmp_path, node) for node in policy._REQUIRED])
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("unexpected" in violation for violation in checks.violations)


def test_fast_partition_requires_mpc_calls_to_pass(policy, tmp_path):
    checks = policy._ProfileChecks(_config(tmp_path, FAST_CORE, collectonly=False))
    session = _finish(
        checks, [_item(tmp_path, node) for node in policy._REQUIRED], passed=False
    )
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("required tests did not pass" in issue for issue in checks.violations)


def test_fast_partition_reports_coverage_limits(policy, tmp_path):
    checks = policy._ProfileChecks(_config(tmp_path, FAST_CORE, collectonly=False))
    _finish(checks, [_item(tmp_path, node) for node in policy._REQUIRED])
    lines = []
    checks.pytest_terminal_summary(SimpleNamespace(write_line=lines.append))
    assert lines == [
        "test profile: fast core (runtime and slow excluded; not full qualification); "
        "native, Lighteval and Phase4 task qualifications excluded"
    ]


@pytest.mark.parametrize(
    "profile,node,reason,present,allowed",
    [
        ("base", "module", "exact", False, True),
        ("base", "module", "changed", False, False),
        ("base", "function", "exact", False, False),
        ("base", "other", "exact", False, False),
        ("base", "module", "exact", True, False),
        ("mandatory", "module", "exact", False, False),
    ],
)
def test_base_dp_skip_exception_remains_exact(
    policy, tmp_path, monkeypatch, profile, node, reason, present, allowed
):
    checks = policy._ProfileChecks(_config(tmp_path, "not runtime", profile=profile))
    nodeid = {
        "module": policy._DP_MODULE,
        "function": policy._DP_MODULE + "::test_case",
        "other": "tests/test_other.py",
    }[node]
    text = policy._DP_REASON if reason == "exact" else "changed missing dependency"
    monkeypatch.setattr(
        policy.importlib.util, "find_spec", lambda name: object() if present else None
    )
    checks.pytest_collectreport(
        SimpleNamespace(nodeid=nodeid, skipped=True, longrepr=("file", 1, text))
    )
    assert bool(checks.allowed_skips) is allowed
    assert bool(checks.violations) is not allowed


@pytest.mark.parametrize("collectonly", [False, True])
def test_dedicated_retained_has_its_own_complete_boundary(
    policy, tmp_path, collectonly
):
    checks = policy._ProfileChecks(
        _config(
            tmp_path,
            "retained_model_search",
            args=[RETAINED_MODULE],
            collectonly=collectonly,
        )
    )
    session = _finish(checks, _retained(tmp_path))
    assert session.exitstatus == pytest.ExitCode.OK, checks.violations
    assert not checks.violations


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "extra",
        "not-passed",
        "deselected",
        "missing-runtime-marker",
    ],
)
def test_dedicated_retained_cannot_drop_or_duplicate_cases(policy, tmp_path, damage):
    checks = policy._ProfileChecks(
        _config(
            tmp_path,
            "retained_model_search",
            args=[RETAINED_MODULE],
            collectonly=damage != "not-passed",
        )
    )
    items = _retained(tmp_path)
    if damage in {"missing", "deselected"}:
        removed = items.pop()
        if damage == "deselected":
            checks.pytest_deselected([removed])
    elif damage == "duplicate":
        items.append(items[0])
    elif damage == "extra":
        items.append(
            _item(
                tmp_path,
                RETAINED_MODULE + "::test_extra",
                "runtime",
                "retained_model_search",
            )
        )
    elif damage == "missing-runtime-marker":
        items[0] = _item(tmp_path, items[0].nodeid, "retained_model_search")
        with pytest.raises(
            pytest.UsageError, match="retained cases require both markers"
        ):
            checks.pytest_collection_modifyitems(items)
        return
    session = _finish(checks, items, passed=damage != "not-passed")
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert checks.violations


@pytest.mark.parametrize(
    "options",
    [
        {"keyword": "not anycostfl"},
        {"ignore": ["tests/x.py"]},
        {"ignore_glob": ["*x*"]},
        {"deselect": ["tests/x.py::test_x"]},
    ],
)
def test_dedicated_retained_rejects_filters(policy, tmp_path, options):
    with pytest.raises(
        pytest.UsageError, match="retained qualification does not permit filters"
    ):
        policy._ProfileChecks(
            _config(
                tmp_path, "retained_model_search", args=[RETAINED_MODULE], **options
            )
        )


@pytest.mark.parametrize(
    "args",
    [
        [RETAINED_MODULE + "::test_sysheterofl_subnet_round"],
        [RETAINED_MODULE, RETAINED_MODULE],
    ],
)
def test_dedicated_retained_requires_the_single_whole_module(policy, tmp_path, args):
    with pytest.raises(pytest.UsageError, match="single complete module"):
        policy._ProfileChecks(_config(tmp_path, "retained_model_search", args=args))


@pytest.mark.parametrize("outcome", ["skip", "xfail", "xpass"])
def test_dedicated_retained_rejects_nonpassing_outcomes(policy, tmp_path, outcome):
    checks = policy._ProfileChecks(
        _config(
            tmp_path, "retained_model_search", args=[RETAINED_MODULE], collectonly=False
        )
    )
    nodeid = _retained(tmp_path)[0].nodeid
    report = SimpleNamespace(
        nodeid=nodeid,
        skipped=outcome != "xpass",
        passed=outcome == "xpass",
        when="call",
        longrepr=("file", 1, "Skipped: injected"),
    )
    if outcome != "skip":
        report.wasxfail = "injected"
    checks.pytest_runtest_logreport(report)
    session = _finish(checks, _retained(tmp_path))
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("unexpected" in violation for violation in checks.violations)


def test_dedicated_keeps_all_mandatory_prerequisite_imports(
    policy, tmp_path, monkeypatch
):
    checks = policy._ProfileChecks(
        _config(tmp_path, "retained_model_search", args=[RETAINED_MODULE])
    )
    imports = []
    monkeypatch.setattr(
        policy.importlib, "import_module", lambda name: imports.append(name)
    )
    checks.pytest_sessionstart()
    assert imports == ["opacus", "kazoo", "gymnasium"]


@pytest.mark.parametrize("collectonly", [False, True])
def test_native_inventory_remains_strict(policy, tmp_path, collectonly):
    checks = policy._ProfileChecks(
        _config(
            tmp_path,
            "",
            args=["tests/mlx_native"],
            profile="mlx-native",
            collectonly=collectonly,
        )
    )
    checks.native_prerequisite_passed = True
    nodeid = "tests/mlx_native/test_native.py::test_expected"
    checks.native_expected[nodeid] = 1
    session = _finish(checks, [])
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("native ledger missing" in violation for violation in checks.violations)


@pytest.fixture(scope="module")
def ci_verifier():
    """Execute the exact independent inline CI verifier without its entrypoint."""
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/pytorch_qualification.yml"
    ).read_text()
    step = workflow.split("- name: Retained model-search qualification", 1)[1]
    source = textwrap.dedent(
        step.split("python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    )
    tree = ast.parse(source)
    assert isinstance(tree.body[-1], ast.Raise)
    namespace = {"__name__": "isolated_ci_verifier"}
    exec(
        compile(
            ast.Module(body=tree.body[:-1], type_ignores=[]),
            "retained-ci-verifier",
            "exec",
        ),
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "missing-collection",
        "duplicate-collection",
        "extra-collection",
        "missing-junit",
        "duplicate-junit",
        "skip",
        "failure",
        "error",
        "xfail",
        "xpass",
    ],
)
def test_independent_ci_gate_checks_collection_and_actual_outcomes(
    ci_verifier, tmp_path, damage
):
    names = list(RETAINED_NAMES)
    collection = tmp_path / "collection.log"
    collected = [f"{RETAINED_MODULE}::{name}" for name in names]
    if damage == "missing-collection":
        collected.pop()
    elif damage == "duplicate-collection":
        collected.append(collected[0])
    elif damage == "extra-collection":
        collected.append("tests/test_other.py::test_other")
    collection.write_text("\n".join(collected) + "\n19 tests collected\n")
    suites = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(
        suites, "testsuite", tests="19", failures="0", errors="0", skipped="0"
    )
    if damage == "missing-junit":
        names.pop()
    elif damage == "duplicate-junit":
        names.append(names[0])
    for index, name in enumerate(names):
        case = ElementTree.SubElement(
            suite,
            "testcase",
            name=name,
            classname="tests.integration.test_retained_model_search",
        )
        if index == 0 and damage in {"skip", "failure", "error", "xfail"}:
            tag = "skipped" if damage in {"skip", "xfail"} else damage
            ElementTree.SubElement(case, tag)
    junit = tmp_path / "junit.xml"
    ElementTree.ElementTree(suites).write(junit)
    runlog = tmp_path / "run.log"
    runlog.write_text("18 passed, 1 xpassed\n" if damage == "xpass" else "19 passed\n")
    receipt = tmp_path / "acceptance.json"
    status = ci_verifier.verify_model_search(collection, junit, runlog, receipt)
    assert status == (0 if damage == "none" else 1)
    record = json.loads(receipt.read_text())
    assert record["accepted"] is (damage == "none")


@pytest.mark.parametrize("missing", [False, True])
def test_stager_requires_all_retained_records_and_keeps_runtime_evidence(
    tmp_path, monkeypatch, missing
):
    path = Path(__file__).parents[1] / ".github/scripts/stage_ci_artifacts.py"
    spec = importlib.util.spec_from_file_location("retained_ci_stager", path)
    assert spec is not None and spec.loader is not None
    stager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stager)
    records = {
        "model-search-packages.txt",
        "model-search-prerequisites.log",
        "model-search-collection.log",
        "model-search.log",
        "model-search.xml",
        "model-search-acceptance.json",
    }
    assert records <= set(stager.EXPECTED_RECORDS)
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    for name in stager.EXPECTED_RECORDS:
        (source / name).write_text("retained evidence\n")
    (source / "model-search-tmp").mkdir()
    (source / "model-search-tmp" / "payload.bin").write_bytes(b"real payload")
    if missing:
        (source / "model-search-acceptance.json").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--require-complete",
        ],
    )
    assert stager.main() == int(missing)
    manifest = json.loads((destination / stager.MANIFEST).read_text())
    assert manifest["missing_expected_records"] == (
        ["model-search-acceptance.json"] if missing else []
    )
    assert (
        destination / "model-search-tmp/payload.bin"
    ).read_bytes() == b"real payload"
