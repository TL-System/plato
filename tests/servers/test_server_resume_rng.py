"""Checkpoint RNG continuation through the synchronous server entrypoint."""

import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import SafetensorError

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from tests.integration.utils import build_minimal_config, configure_environment

POPULATION = list(range(1, 11))


@pytest.fixture(autouse=True)
def preserve_rng_states():
    """Keep these deterministic probes from changing other tests' RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def runtime_config(central, seed):
    config = build_minimal_config(total_clients=10, clients_per_round=2)
    config["server"].update(do_test=False, disable_clients=False)
    if seed is None:
        del config["server"]["random_seed"]
    else:
        config["server"]["random_seed"] = seed
    if central:
        config["algorithm"].update(cross_silo=True, total_silos=2, local_rounds=1)
    return config


def make_server(server_type=None):
    """Use real configuration, trainer, checkpoint and selection implementations."""
    from plato.servers import fedavg

    server_type = server_type or fedavg.Server
    return server_type(model=lambda: torch.nn.Linear(2, 1), trainer=ComposableTrainer)


def intercept_process_launches(monkeypatch):
    from plato.servers import base

    launches = []
    monkeypatch.setattr(
        base.Server, "_start_clients", staticmethod(lambda **kw: launches.append(kw))
    )
    return launches


def assert_launch_branch(launches, central):
    assert len(launches) == 1
    assert launches[0].get("as_server", False) == central


def assert_numpy_state_equal(expected):
    actual = np.random.get_state()
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def save_committed_round(unrelated_draws=0):
    server = make_server()
    server.configure()
    random.seed(17)
    np.random.seed(19)
    np.random.random(5)
    server.prng_state = random.getstate()
    prior = [server._select_clients_with_strategy(POPULATION, 2) for _ in range(2)]
    assert prior == [[9, 7], [5, 6]]
    server.current_round = 2
    with torch.no_grad():
        server.trainer.model.weight.copy_(torch.tensor([[1.25, -2.5]]))
        server.trainer.model.bias.copy_(torch.tensor([0.75]))
    weights = {
        name: value.clone() for name, value in server.trainer.model.state_dict().items()
    }
    python_reference = random.Random()
    python_reference.setstate(server.prng_state)
    numpy_reference = np.random.RandomState()
    numpy_reference.set_state(np.random.get_state())
    for _ in range(unrelated_draws):
        random.random()
    global_python_state = random.getstate()
    selection_state = server.prng_state
    context_state = server.context.state["prng_state"]
    if unrelated_draws:
        assert global_python_state != selection_state
    server.save_to_checkpoint()
    # Writing a checkpoint must not reset either live Python stream or NumPy.
    assert random.getstate() == global_python_state
    assert server.prng_state == selection_state
    assert server.context.state["prng_state"] == context_state
    assert_numpy_state_equal(numpy_reference.get_state())
    assert {path.name for path in Path(Config.params["checkpoint_path"]).iterdir()} == {
        "checkpoint_lenet5_2.safetensors",
        "checkpoint_lenet5_2.safetensors.pkl",
        "current_round.pkl",
        "numpy_prng_state_2.pkl",
        "prng_state_2.pkl",
    }
    uninterrupted = [select_and_check_state(server) for _ in range(3)]
    return weights, python_reference, numpy_reference, uninterrupted


def select_and_check_state(server):
    selected = server._select_clients_with_strategy(POPULATION, 2)
    assert server.prng_state == server.context.state["prng_state"]
    assert server.prng_state == random.getstate()
    assert server.context.current_round == server.current_round
    return selected


@pytest.mark.parametrize("central", [False, True], ids=["ordinary", "central"])
@pytest.mark.parametrize("seed", [17, None], ids=["configured-seed", "no-seed"])
@pytest.mark.parametrize("unrelated_draws", [0, 1, 13], ids=["zero", "one", "many"])
def test_actual_checkpoint_continues_selection_through_run(
    monkeypatch, tmp_path, central, seed, unrelated_draws
):
    with configure_environment(runtime_config(central, seed), runtime_root=tmp_path):
        weights, python_reference, numpy_reference, uninterrupted = (
            save_committed_round(unrelated_draws)
        )
        expected = [python_reference.sample(POPULATION, 2) for _ in range(3)]
        assert expected[0] == [5, 3]
        assert uninterrupted == expected
        random.seed(99)
        np.random.seed(98)
        server = make_server()
        launches = intercept_process_launches(monkeypatch)
        Config.args.resume = True
        observed = []

        def capture_start():
            assert server.current_round == 2
            assert server.resumed_session
            assert server.prng_state == random.getstate()
            assert_numpy_state_equal(numpy_reference.get_state())
            for name, value in server.trainer.model.state_dict().items():
                assert torch.equal(value, weights[name])
            for _ in range(3):
                observed.append(select_and_check_state(server))
                assert np.random.random() == numpy_reference.random_sample()

        # Replace only the external socket listener and child-process launches.
        monkeypatch.setattr(server, "start", capture_start)
        server.run()
        assert_launch_branch(launches, central)
        assert observed == expected
        assert server.prng_state == python_reference.getstate()


@pytest.mark.parametrize("central", [False, True], ids=["ordinary", "central"])
def test_legacy_global_rng_tuple_remains_readable(monkeypatch, tmp_path, central):
    with configure_environment(runtime_config(central, 17), runtime_root=tmp_path):
        weights, reference, numpy_reference, uninterrupted = save_committed_round()
        # Legacy writers captured global Python state after unrelated draws.
        # Its lost selection boundary cannot be reconstructed from this tuple.
        reference.random()
        legacy_state = reference.getstate()
        path = Path(Config.params["checkpoint_path"]) / "prng_state_2.pkl"
        with path.open("wb") as stream:
            pickle.dump(legacy_state, stream, protocol=4)
        expected = [reference.sample(POPULATION, 2) for _ in range(3)]
        assert expected != uninterrupted
        random.seed(99)
        np.random.seed(98)
        server = make_server()
        launches = intercept_process_launches(monkeypatch)
        Config.args.resume = True
        observed = []

        def capture_start():
            assert server.current_round == 2
            assert server.resumed_session
            assert random.getstate() == server.prng_state == legacy_state
            assert_numpy_state_equal(numpy_reference.get_state())
            for name, value in server.trainer.model.state_dict().items():
                assert torch.equal(value, weights[name])
            for _ in range(3):
                observed.append(select_and_check_state(server))
                assert np.random.random() == numpy_reference.random_sample()

        monkeypatch.setattr(server, "start", capture_start)
        server.run()
        assert_launch_branch(launches, central)
        assert observed == expected
        assert server.prng_state == random.getstate() == reference.getstate()


@pytest.mark.parametrize("central", [False, True], ids=["ordinary", "central"])
@pytest.mark.parametrize("seed", [17, None], ids=["configured-seed", "no-seed"])
def test_fresh_run_keeps_configured_seed_or_initial_selection_state(
    monkeypatch, tmp_path, central, seed
):
    with configure_environment(runtime_config(central, seed), runtime_root=tmp_path):
        random.seed(31)
        np.random.seed(29)
        numpy_reference = np.random.RandomState()
        numpy_reference.set_state(np.random.get_state())
        server = make_server()
        reference = random.Random()
        reference.setstate(server.prng_state)
        if seed is not None:
            reference.seed(seed)
        expected = [reference.sample(POPULATION, 2) for _ in range(3)]
        # A no-seed run uses the server's captured selection state, even if other
        # server setup consumes unrelated global Python randomness.
        random.random()
        launches = intercept_process_launches(monkeypatch)
        observed = []

        def capture_start():
            assert server.current_round == 0
            assert not server.resumed_session
            for _ in range(3):
                observed.append(select_and_check_state(server))
                assert np.random.random() == numpy_reference.random_sample()

        monkeypatch.setattr(server, "start", capture_start)
        server.run()
        assert_launch_branch(launches, central)
        assert observed == expected
        assert server.prng_state == reference.getstate()


@pytest.mark.parametrize("central", [False, True], ids=["ordinary", "central"])
def test_custom_resume_override_owns_selection_state_without_session_flag(
    monkeypatch, tmp_path, central
):
    reference = random.Random(23)
    reference.sample(POPULATION, 2)
    restored_state = reference.getstate()
    expected = [reference.sample(POPULATION, 2) for _ in range(3)]

    with configure_environment(runtime_config(central, 17), runtime_root=tmp_path):
        from plato.servers import fedavg

        class CustomResumeServer(fedavg.Server):
            def _resume_from_checkpoint(self):
                self.current_round = 4
                self.prng_state = restored_state
                random.setstate(restored_state)
                # Existing custom overrides need not set base.resumed_session.

        server = make_server(CustomResumeServer)
        launches = intercept_process_launches(monkeypatch)
        Config.args.resume = True
        observed = []

        def capture_start():
            assert server.current_round == 4
            assert not server.resumed_session
            assert random.getstate() == restored_state
            observed.extend(select_and_check_state(server) for _ in range(3))

        monkeypatch.setattr(server, "start", capture_start)
        server.run()
        assert_launch_branch(launches, central)
        assert observed == expected


@pytest.mark.parametrize("central", [False, True], ids=["ordinary", "central"])
@pytest.mark.parametrize(
    "filename,error",
    [
        ("current_round.pkl", pickle.UnpicklingError),
        ("numpy_prng_state_2.pkl", pickle.UnpicklingError),
        ("prng_state_2.pkl", pickle.UnpicklingError),
        ("checkpoint_lenet5_2.safetensors", SafetensorError),
    ],
)
def test_malformed_resume_propagates_before_process_or_socket_start(
    monkeypatch, tmp_path, central, filename, error
):
    with configure_environment(runtime_config(central, 17), runtime_root=tmp_path):
        save_committed_round()
        (Path(Config.params["checkpoint_path"]) / filename).write_bytes(b"broken")
        server = make_server()
        launches = intercept_process_launches(monkeypatch)
        Config.args.resume = True
        external_starts = []
        monkeypatch.setattr(server, "start", lambda: external_starts.append("socket"))
        monkeypatch.setattr(
            server, "_periodic", lambda _: external_starts.append("periodic")
        )
        with pytest.raises(error):
            server.run()
        assert launches == []
        assert external_starts == []
