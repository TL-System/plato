"""Exercise the production collection policy in bounded, isolated pytest projects."""

import ast
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass(frozen=True)
class _PytestResult:
    """Keep the bounded subprocess status and combined diagnostic output."""

    returncode: int
    output: str


@pytest.fixture(scope="module")
def profile_source():
    """Copy exact production hooks, excluding unrelated heavyweight fixtures."""
    source = (Path(__file__).parent / "conftest.py").read_text()
    nodes = ast.parse(source).body
    allowed_imports = {
        "importlib",
        "json",
        "math",
        "platform",
        "re",
        "collections",
        "pathlib",
        "pytest",
    }
    segments = []
    for node in nodes:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.decorator_list
        ):
            if any("fixture" in ast.unparse(item) for item in node.decorator_list):
                break
        if isinstance(node, ast.Import):
            if not all(
                item.name.split(".")[0] in allowed_imports for item in node.names
            ):
                continue
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".")[0] not in allowed_imports:
                continue
        elif not isinstance(
            node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef)
        ):
            continue
        start = min(
            [
                node.lineno,
                *(item.lineno for item in getattr(node, "decorator_list", [])),
            ]
        )
        segments.append("\n".join(source.splitlines()[start - 1 : node.end_lineno]))
    return "\n\n".join(segments) + "\n"


@pytest.fixture(scope="module")
def policy(profile_source):
    namespace = {"__name__": "isolated_plato_profile"}
    exec(compile(profile_source, "tests/conftest.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def project(tmp_path, profile_source, policy):
    """Use real hooks, test-local native preflight, and an independent tiny ledger."""
    tests = tmp_path / "tests"
    native = tests / "mlx_native"
    native.mkdir(parents=True)
    (tests / "__init__.py").write_text('"""Inert test package."""\n')
    (native / "__init__.py").write_text('"""Inert native package."""\n')
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\naddopts = --strict-markers\n"
        "markers =\n    runtime: core runtime partition\n    mlx_native: native profile\n"
    )
    for name in ("opacus", "kazoo", "gymnasium"):
        (tmp_path / f"{name}.py").write_text(
            '"""Policy harness dependency stand-in."""\n'
        )
    # Preserve the actual core ledger checks even in the miniature project.
    for nodeid in policy._REQUIRED:
        path, name = nodeid.split("::")
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as stream:
            stream.write(f"def {name}():\n    pass\n\n")
    startup = tmp_path / policy._STARTUP.split("::")[0]
    startup.write_text("import pytest\npytestmark = pytest.mark.runtime\n")
    for name, count in policy._STARTUP_CASES.items():
        with startup.open("a") as stream:
            if policy._STARTUP + name in policy._RUNTIME:
                stream.write("@pytest.mark.runtime\n")
            if count == 1:
                stream.write(f"def {name}():\n    pass\n\n")
            else:
                stream.write(
                    f"@pytest.mark.parametrize('case', range({count}))\n"
                    f"def {name}(case):\n    pass\n\n"
                )
    (tests / "test_core.py").write_text("def test_core():\n    pass\n")
    (tests / "test_mlx_native_suffix.py").write_text(
        "def test_core_name():\n    pass\n"
    )
    sentinel = (
        "from pathlib import Path\n"
        "with (Path(__file__).parents[2] / 'events').open('a') as stream:\n"
        "    stream.write('native-import\\n')\n"
    )
    (native / "test_native.py").write_text(
        sentinel + "import pytest\n"
        "@pytest.mark.parametrize('value', [1, 2], ids=['one', 'two'])\n"
        "def test_native(value):\n    assert value > 0\n"
    )
    cases = [
        {
            "nodeid": f"tests/mlx_native/test_native.py::test_native[{name}]",
            "criteria": ["E1"],
        }
        for name in ("one", "two")
    ]
    (native / "cases.json").write_text(
        json.dumps({"schema_version": 1, "cases": cases})
    )
    hooks = (
        "\n# Test-local simulation; this project never claims native support.\n"
        "def _native_prerequisites():\n"
        "    with (Path(__file__).parents[1] / 'events').open('a') as stream:\n"
        "        stream.write('preflight\\n')\n"
        "def pytest_pycollect_makemodule(module_path, parent):\n"
        "    if 'mlx_native' in module_path.parts:\n"
        "        with (Path(__file__).parents[1] / 'events').open('a') as stream:\n"
        "            stream.write('native-collector\\n')\n"
    )
    (tests / "conftest.py").write_text(profile_source + hooks)
    return SimpleNamespace(
        root=tmp_path,
        tests=tests,
        native=native,
        cases=cases,
        sentinel=sentinel,
        production_source=profile_source,
    )


def run_pytest(project, *arguments, cwd=None, addopts=None) -> _PytestResult:
    """Limit every subprocess and keep its import search inside the tiny project."""
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join([str(project.root), str(project.tests)])
    if addopts is not None:
        environment["PYTEST_ADDOPTS"] = addopts
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *map(str, arguments)],
        cwd=cwd or project.root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return _PytestResult(result.returncode, result.stdout + result.stderr)


def events(project):
    path = project.root / "events"
    return path.read_text().splitlines() if path.exists() else []


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["tests"],
        ["."],
        ["tests", "--test-profile=base"],
        ["tests", "--test-profile=mandatory"],
        ["--pyargs", "tests"],
        ["--pyargs", "tests.test_core"],
        ["--pyargs", "tests.test_mlx_native_suffix"],
    ],
)
def test_core_scope_excludes_native_before_import(project, arguments):
    result = run_pytest(project, *arguments)
    assert result.returncode == 0, result.output
    assert events(project) == []
    assert "core scope (native excluded)" in result.output


@pytest.mark.parametrize(
    "target",
    [
        "tests/mlx_native",
        "tests/mlx_native/test_native.py",
        "tests/mlx_native/test_native.py::test_native[one]",
    ],
)
@pytest.mark.parametrize("profile", ["base", "mandatory"])
def test_explicit_native_conflicts_fail_before_import(project, target, profile):
    result = run_pytest(project, target, f"--test-profile={profile}")
    assert result.returncode != 0
    assert "mlx-native target conflicts" in result.output
    assert events(project) == []


@pytest.mark.parametrize(
    "target",
    [
        "tests.mlx_native",
        "tests.mlx_native.test_native",
        "tests.mlx_native.test_native::test_native[one]",
    ],
)
@pytest.mark.parametrize(
    "options",
    [[], ["--test-profile=base"], ["--test-profile=mandatory"], ["--collect-only"]],
)
def test_canonical_native_pyargs_rejected_before_import(project, target, options):
    result = run_pytest(project, "--pyargs", target, *options)
    assert result.returncode != 0
    assert "mlx-native selector unsupported" in result.output
    assert "filesystem paths" in result.output
    assert events(project) == []


def test_alternative_pyargs_route_is_guarded_before_collector(project):
    result = run_pytest(
        project, "--pyargs", "mlx_native.test_native", cwd=project.tests
    )
    assert result.returncode != 0
    assert "mlx-native selector unsupported" in result.output
    assert events(project) == []


def test_bare_native_module_alias_is_guarded_before_import(project):
    result = run_pytest(project, "--pyargs", "test_native", cwd=project.native)
    assert result.returncode != 0
    assert "mlx-native selector unsupported" in result.output
    assert events(project) == []


def test_native_filesystem_target_with_pyargs_is_rejected_early(project):
    result = run_pytest(project, "--pyargs", "tests/mlx_native/test_native.py")
    assert result.returncode != 0
    assert "mlx-native selector unsupported" in result.output
    assert events(project) == []


@pytest.mark.parametrize("form", ["relative", "absolute", "from-tests", "mixed"])
def test_focused_native_filesystem_preflight_precedes_import(project, form):
    target = "tests/mlx_native/test_native.py::test_native[one]"
    cwd = None
    arguments = [target]
    if form == "absolute":
        arguments = [str(project.root / target)]
    elif form == "from-tests":
        cwd, arguments = project.tests, ["mlx_native/test_native.py::test_native[one]"]
    elif form == "mixed":
        arguments = ["tests/test_core.py", target]
    result = run_pytest(project, *arguments, cwd=cwd)
    assert result.returncode == 0, result.output
    observed = events(project)
    assert observed[0] == "preflight"
    assert observed.index("native-collector") < observed.index("native-import")
    assert "native focused execution (not complete qualification)" in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["-k", "native"],
        ["-m", "mlx_native"],
        ["--deselect", "x"],
        ["--ignore", "x"],
        ["--ignore-glob", "*x*"],
        ["--pyargs"],
    ],
)
@pytest.mark.parametrize("injected", [False, True])
def test_native_qualification_rejects_effective_filters(project, arguments, injected):
    result = run_pytest(
        project,
        "tests/mlx_native",
        "--test-profile=mlx-native",
        *(arguments if not injected else []),
        addopts=" ".join(arguments) if injected else None,
    )
    assert result.returncode != 0
    assert "mlx-native" in result.output
    assert events(project) == []


def test_native_qualification_rejects_ini_addopts_filters(project):
    configuration = project.root / "pytest.ini"
    configuration.write_text(
        configuration.read_text().replace(
            "addopts = --strict-markers", "addopts = --strict-markers -k native"
        )
    )
    result = run_pytest(project, "tests/mlx_native", "--test-profile=mlx-native")
    assert result.returncode != 0
    assert "mlx-native qualification does not permit" in result.output
    assert events(project) == []


def test_marker_only_native_request_is_diagnostic(project):
    result = run_pytest(project, "tests", "-m", "mlx_native")
    assert result.returncode != 0
    assert "mlx_native marker requires" in result.output
    assert events(project) == []


@pytest.mark.parametrize("collect", [False, True])
def test_exact_native_ledger_succeeds_with_honest_summary(project, collect):
    result = run_pytest(
        project,
        "tests/mlx_native",
        "--test-profile=mlx-native",
        *(["--collect-only"] if collect else []),
    )
    assert result.returncode == 0, result.output
    expected = (
        "native collection-only validation"
        if collect
        else "native complete qualification"
    )
    assert expected in result.output
    assert ("native complete qualification" in result.output) is not collect


@pytest.mark.parametrize("damage", ["missing", "extra", "parameter", "duplicate"])
def test_changed_native_collection_cannot_qualify(project, damage):
    module = project.native / "test_native.py"
    source = module.read_text()
    if damage == "missing":
        module.write_text(
            source.replace("[1, 2]", "[1]").replace("['one', 'two']", "['one']")
        )
    elif damage == "extra":
        module.write_text(source + "\ndef test_unlisted():\n    pass\n")
    elif damage == "parameter":
        module.write_text(source.replace("'two'", "'renamed'"))
    else:
        with (project.tests / "conftest.py").open("a") as stream:
            stream.write(
                "\ndef pytest_collection_modifyitems(items):\n    items.append(items[0])\n"
            )
    result = run_pytest(
        project, "tests/mlx_native", "--test-profile=mlx-native", "--collect-only"
    )
    assert result.returncode != 0
    assert "native ledger" in result.output
    assert "native complete qualification" not in result.output


@pytest.mark.parametrize(
    "ledger",
    [
        {"schema_version": 1, "cases": []},
        {"schema_version": 1, "cases": [{"nodeid": "other::test", "criteria": ["E1"]}]},
        {
            "schema_version": 1,
            "cases": [{"nodeid": "tests/mlx_native/x.py::test", "criteria": ["E8"]}],
        },
        {"schema_version": 2, "cases": []},
    ],
)
def test_invalid_native_ledger_fails_clearly(project, ledger):
    (project.native / "cases.json").write_text(json.dumps(ledger))
    result = run_pytest(project, "tests/mlx_native", "--test-profile=mlx-native")
    assert result.returncode != 0
    assert "mlx-native ledger invalid" in result.output


def test_duplicate_native_ledger_fails_clearly(project):
    ledger = {"schema_version": 1, "cases": [project.cases[0], project.cases[0]]}
    (project.native / "cases.json").write_text(json.dumps(ledger))
    result = run_pytest(project, "tests/mlx_native", "--test-profile=mlx-native")
    assert result.returncode != 0
    assert "duplicate" in result.output


@pytest.mark.parametrize(
    "outcome",
    [
        "skip",
        "collect-skip",
        "xfail",
        "strict-xfail",
        "xpass",
        "setup-failure",
        "teardown-failure",
    ],
)
@pytest.mark.parametrize("qualified", [False, True])
def test_native_outcomes_never_disappear_behind_success(project, outcome, qualified):
    module = project.native / "test_native.py"
    source = module.read_text()
    if outcome == "collect-skip":
        source = (
            project.sentinel
            + "import pytest\npytest.skip('native collection skip', allow_module_level=True)\n"
        )
    elif outcome in {"skip", "xfail", "strict-xfail", "xpass"}:
        if outcome == "skip":
            source = source.replace("assert value > 0", "pytest.skip('native skip')")
        else:
            source = source.replace(
                "@pytest.mark.parametrize",
                f"@pytest.mark.xfail(strict={outcome == 'strict-xfail'})\n@pytest.mark.parametrize",
            )
            if outcome != "xpass":
                source = source.replace("assert value > 0", "assert False")
    else:
        fixture = (
            "assert False\n    yield"
            if outcome == "setup-failure"
            else "yield\n    assert False"
        )
        source += f"\n@pytest.fixture(autouse=True)\ndef broken():\n    {fixture}\n"
    module.write_text(source)
    options = ["--test-profile=mlx-native"] if qualified else []
    result = run_pytest(project, "tests/mlx_native", *options)
    assert result.returncode != 0
    assert "native complete qualification" not in result.output
    if outcome in {"skip", "collect-skip", "xfail", "strict-xfail", "xpass"}:
        assert "unexpected" in result.output


def test_subset_native_profile_requires_entire_ledger(project):
    result = run_pytest(
        project,
        "tests/mlx_native/test_native.py::test_native[one]",
        "--test-profile=mlx-native",
    )
    assert result.returncode != 0
    assert "native ledger missing" in result.output


def test_native_marker_outside_boundary_is_layout_error(project):
    (project.tests / "test_core.py").write_text(
        "import pytest\n@pytest.mark.mlx_native\ndef test_core():\n    pass\n"
    )
    result = run_pytest(project, "tests/test_core.py")
    assert result.returncode != 0
    assert "mlx_native marker outside" in result.output


@pytest.mark.parametrize("partition", ["runtime", "not runtime"])
@pytest.mark.parametrize("profile", ["base", "mandatory"])
def test_existing_core_runtime_partitions_keep_their_ledger(
    project, partition, profile
):
    result = run_pytest(project, "tests", f"--test-profile={profile}", "-m", partition)
    assert result.returncode == 0, result.output
    assert events(project) == []


def test_existing_core_collection_exclusions_remain_failures(project):
    result = run_pytest(
        project, "tests", "--test-profile=base", "--ignore", "tests/test_core.py"
    )
    assert result.returncode != 0
    assert "test profiles do not permit collection exclusions" in result.output


def test_native_profile_with_core_only_target_cannot_qualify(project):
    result = run_pytest(project, "tests/test_core.py", "--test-profile=mlx-native")
    assert result.returncode != 0
    assert "native ledger missing" in result.output
    assert "native complete qualification" not in result.output


def test_native_pyargs_profile_rejects_ordinary_core_target(project):
    result = run_pytest(
        project, "--pyargs", "tests.test_core", "--test-profile=mlx-native"
    )
    assert result.returncode != 0
    assert "mlx-native selector unsupported" in result.output
    assert events(project) == []


def test_native_ledger_invalid_json_is_scoped(project):
    (project.native / "cases.json").write_text("{invalid")
    result = run_pytest(project, "tests/mlx_native", "--test-profile=mlx-native")
    assert result.returncode != 0
    assert "mlx-native ledger invalid" in result.output


def test_native_profile_duplicate_explicit_file_cannot_qualify(project):
    result = run_pytest(
        project,
        "tests/mlx_native/test_native.py",
        "tests/mlx_native/test_native.py",
        "--keep-duplicates",
        "--test-profile=mlx-native",
        "--collect-only",
    )
    assert result.returncode != 0
    assert "native ledger duplicate collection" in result.output


def test_actual_missing_mlx_guard_rejects_before_module_import(project):
    # Replace only the scratch copy with unmodified production hooks.
    (project.tests / "conftest.py").write_text(project.production_source)
    (project.root / "mlx").mkdir()
    (project.root / "mlx" / "__init__.py").write_text(
        '"""Deliberately no native backend."""\n'
    )
    result = run_pytest(project, "tests/mlx_native/test_native.py")
    assert result.returncode != 0
    assert "mlx-native prerequisite failed" in result.output
    assert events(project) == []


def test_unpreflighted_file_wrapper_rejects_before_yield(policy, tmp_path):
    boundary = tmp_path / "tests/mlx_native"
    config = SimpleNamespace(
        rootpath=tmp_path,
        invocation_params=SimpleNamespace(dir=tmp_path),
        args=[str(boundary)],
    )
    options = {
        "test_profile": None,
        "keyword": "",
        "markexpr": "",
        "deselect": [],
        "pyargs": False,
    }
    config.getoption = lambda name: options.get(name)
    checks = policy._ProfileChecks(config)
    assert checks.native_requested
    assert not checks.native_prerequisite_passed
    wrapper = checks.pytest_collect_file(
        boundary / "test_native.py", SimpleNamespace(config=config)
    )
    with pytest.raises(pytest.UsageError, match="mlx-native preflight required"):
        next(wrapper)


def test_native_manual_configure_does_not_probe(policy, tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("configure must never initialize native backend")

    monkeypatch.setitem(
        policy._ProfileChecks.__init__.__globals__, "_native_prerequisites", forbidden
    )
    config = pytest.Config.fromdictargs(
        {}, [str(tmp_path / "tests/mlx_native"), "-p", "no:cacheprovider"]
    )
    config.option.test_profile = "mlx-native"
    try:
        policy.pytest_configure(config)
        checks = config.pluginmanager.get_plugin("plato-profile-checks")
        assert checks is not None
        assert checks.qualify and not checks.base and checks.native_requested
        assert not checks.native_prerequisite_passed
    finally:
        config._ensure_unconfigure()


@pytest.fixture
def simulated_native(policy, monkeypatch):
    """Model public stream restoration and injected native prerequisite failures."""
    state = SimpleNamespace(
        active="caller",
        evaluated=[],
        synchronized=[],
        failure=None,
        available=True,
        result=3.0,
        versions={"mlx": "1.2.3", "mlx-metal": "1.2.3"},
    )

    @contextmanager
    def stream(device):
        previous = state.active
        state.active = device
        try:
            yield
        finally:
            state.active = previous

    def evaluate(_value):
        state.evaluated.append(state.active)
        if state.failure == state.active:
            raise RuntimeError(f"{state.active} calculation failed")

    core = SimpleNamespace(
        cpu="cpu",
        gpu="gpu",
        float32="float32",
        stream=stream,
        metal=SimpleNamespace(is_available=lambda: state.available),
        array=lambda values, dtype: values,
        sum=lambda _values: SimpleNamespace(item=lambda: state.result),
        eval=evaluate,
        synchronize=lambda _stream=None: state.synchronized.append(state.active),
    )
    modules = {"mlx.core": core, "mlx.nn": object(), "mlx.optimizers": object()}
    imports = SimpleNamespace(
        import_module=lambda name: modules[name],
        metadata=SimpleNamespace(version=lambda name: state.versions[name]),
    )
    globals_ = policy._native_prerequisites.__globals__
    monkeypatch.setitem(globals_, "importlib", imports)
    monkeypatch.setitem(
        globals_,
        "platform",
        SimpleNamespace(system=lambda: "Darwin", machine=lambda: "arm64"),
    )
    return SimpleNamespace(state=state, imports=imports, globals=globals_)


def test_native_prerequisite_forces_cpu_and_gpu_and_restores_streams(
    policy, simulated_native
):
    policy._native_prerequisites()
    assert simulated_native.state.evaluated == ["cpu", "gpu"]
    assert simulated_native.state.synchronized == ["cpu", "gpu"]
    assert simulated_native.state.active == "caller"


@pytest.mark.parametrize(
    "failure", ["cpu", "gpu", "metadata", "import", "metal", "version", "nan", "wrong"]
)
def test_native_prerequisite_failures_are_scoped_and_preserve_cause(
    policy, simulated_native, failure
):
    probe = simulated_native
    if failure in {"cpu", "gpu"}:
        probe.state.failure = failure
    elif failure == "metadata":

        def missing(_name):
            raise LookupError("distribution metadata missing")

        probe.imports.metadata.version = missing
    elif failure == "import":

        def missing(_name):
            raise ModuleNotFoundError("native import missing")

        probe.imports.import_module = missing
    elif failure == "metal":
        probe.state.available = False
    elif failure == "version":
        probe.state.versions["mlx-metal"] = "9.9.9"
    elif failure == "nan":
        probe.state.result = float("nan")
    else:
        probe.state.result = 4.0
    with pytest.raises(
        pytest.UsageError, match="mlx-native prerequisite failed"
    ) as caught:
        policy._native_prerequisites()
    assert caught.value.__cause__ is not None
    assert "Darwin arm64" in str(caught.value)
    assert probe.state.active == "caller"


@pytest.mark.parametrize("system,machine", [("Linux", "aarch64"), ("Darwin", "x86_64")])
def test_native_selected_platform_failure_does_not_import(
    policy, simulated_native, system, machine
):
    simulated_native.globals["platform"] = SimpleNamespace(
        system=lambda: system, machine=lambda: machine
    )

    def forbidden(_name):
        raise AssertionError("unsupported selected platform must fail before import")

    simulated_native.imports.import_module = forbidden
    with pytest.raises(pytest.UsageError, match=f"{system} {machine}"):
        policy._native_prerequisites()


def test_native_prerequisite_preserves_keyboard_interrupt(policy, simulated_native):
    def interrupted(_name):
        raise KeyboardInterrupt()

    simulated_native.imports.import_module = interrupted
    with pytest.raises(KeyboardInterrupt):
        policy._native_prerequisites()
