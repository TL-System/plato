"""Independent merged checkpoint codec and selection continuation probe."""
import copy
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import plato
import torch

from plato.config import Config
from tests.integration.utils import build_minimal_config, configure_environment
from tests.integration.test_checkpoint_runtime_agreement import make_server

version = f"{sys.version_info.major}{sys.version_info.minor}"
scratch = Path(f"/tmp/plato-phase2-root/merged-resume-{version}-cases")
scratch.mkdir(exist_ok=True)
population = list(range(1, 11))
results = []
for central in (False, True):
    for name in ("lenet5", "org/model", "org_model", "x" * 240):
        for draws in (0, 7):
            config = build_minimal_config(
                model_name=name, total_clients=10, clients_per_round=2
            )
            config["server"].update(random_seed=17, disable_clients=False)
            if central:
                config["algorithm"].update(
                    cross_silo=True, total_silos=2, local_rounds=1
                )
            case_root = scratch / f"case-{len(results)}"
            case_root.mkdir(parents=True, exist_ok=True)
            with configure_environment(config, runtime_root=case_root):
                from plato.servers import base
                server = make_server()
                server.disable_clients = False
                server.current_round = 2
                server.trainer.run_history.update_metric("round_marker", 123)
                random.seed(17)
                np.random.seed(19)
                np.random.random(3)
                server.prng_state = random.getstate()
                prior = [
                    server._select_clients_with_strategy(population, 2)
                    for _ in range(2)
                ]
                assert prior == [[9, 7], [5, 6]]
                owned_state = server.prng_state
                reference = random.Random()
                reference.setstate(owned_state)
                expected = [reference.sample(population, 2) for _ in range(5)]
                numpy_reference = np.random.RandomState()
                numpy_reference.set_state(np.random.get_state())
                expected_numpy = numpy_reference.random_sample(3)
                for _ in range(draws):
                    random.random()
                before_global = random.getstate()
                assert (before_global != owned_state) == bool(draws)
                weights = copy.deepcopy(server.trainer.model.state_dict())
                before_numpy = np.random.get_state()
                server.save_to_checkpoint()
                assert random.getstate() == before_global
                assert server.prng_state == owned_state
                after_numpy = np.random.get_state()
                assert before_numpy[0] == after_numpy[0]
                np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
                assert before_numpy[2:] == after_numpy[2:]
                uninterrupted = [
                    server._select_clients_with_strategy(population, 2)
                    for _ in range(5)
                ]
                assert uninterrupted == expected
                with torch.no_grad():
                    for parameter in server.trainer.model.parameters():
                        parameter.zero_()
                server.trainer.run_history.reset()
                server.current_round = 99
                random.seed(99)
                np.random.seed(98)
                server.prng_state = random.getstate()
                Config().args.resume = True
                server.configure = lambda: None

                async def periodic(interval):
                    pass

                server._periodic = periodic
                observed = {}
                launches = []

                def start():
                    actual = [
                        server._select_clients_with_strategy(population, 2)
                        for _ in range(5)
                    ]
                    assert actual == expected
                    assert server.prng_state == server.context.state["prng_state"]
                    assert server.current_round == 2 and server.resumed_session
                    np.testing.assert_array_equal(
                        np.random.random(3), expected_numpy
                    )
                    for key, value in weights.items():
                        torch.testing.assert_close(
                            server.trainer.model.state_dict()[key], value
                        )
                    assert server.trainer.run_history.get_metric_values(
                        "round_marker"
                    ) == [123]
                    observed.update(actual=actual, round=server.current_round)

                server.start = start
                with patch.object(
                    base.Server,
                    "_start_clients",
                    staticmethod(lambda **kwargs: launches.append(kwargs)),
                ):
                    server.run()
                assert len(launches) == 1
                assert bool(launches[0].get("as_server", False)) == central
                root = Path(Config.params["checkpoint_path"])
                files = sorted(path.name for path in root.iterdir())
                assert all(path.parent == root for path in root.iterdir())
                assert max(len(item.encode()) for item in files) <= 255
                results.append({
                    "central": central,
                    "model_name": name,
                    "unrelated_draws": draws,
                    "expected": expected,
                    "observed": observed,
                    "files": files,
                    "save_did_not_mutate_rng": True,
                    "model_round_history_numpy_restored": True,
                })

print(json.dumps({
    "python": sys.version,
    "source": plato.__file__,
    "cases": len(results),
    "passed": len(results),
    "results": results,
    "limits": (
        "Real save/load/run/selection; configure preserves the constructed "
        "model, child launch and network start are intercepted. "
        "Not a resumed network round."
    ),
}, indent=2))
