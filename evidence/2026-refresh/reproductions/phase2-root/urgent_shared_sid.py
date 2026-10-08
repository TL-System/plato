import asyncio
import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.integration.utils import build_minimal_config, configure_environment
from tests.servers.test_runtime_transport import assigned_server, report_bytes

async def scenario():
    server = assigned_server()
    server.clients[100]["client_id"] = 9
    server._session_assignments.clear()
    server.training_clients = {}
    server.training_sids = []
    server.selected_clients = [9]
    server.current_reported_clients = {9: True}
    server.current_round = 3
    server.request_update = server.asynchronous_mode = server.simulate_wall_time = True
    server.reported_clients = [
        (100 + index, client_id, {
            "client_id": client_id, "starting_round": 1, "start_time": 0,
            "sid": "worker", "report": pickle.loads(report_bytes(client_id)),
        })
        for index, client_id in enumerate([7, 2])
    ]
    server.should_request_update = lambda **kwargs: True
    server.sio = SimpleNamespace(emit=AsyncMock())
    await server._process_clients(server.reported_clients[0])
    result = {
        "requests": [{"event": call.args[0], "client_id": call.args[1]["client_id"], "room": call.kwargs["room"]} for call in server.sio.emit.await_args_list],
        "assignments": dict(server._session_assignments),
        "training_clients": list(server.training_clients),
        "training_sids": list(server.training_sids),
    }
    try:
        await server._client_report_arrived("worker", 7, report_bytes(7))
        result["first_response"] = "accepted"
    except Exception as error:
        result["first_response"] = {"error": type(error).__name__, "message": str(error)}
    print("PROBE_RESULT", json.dumps(result))

with tempfile.TemporaryDirectory(prefix="plato-urgent-sid-probe-") as temporary:
    with configure_environment(build_minimal_config(), runtime_root=Path(temporary)):
        asyncio.run(scenario())
