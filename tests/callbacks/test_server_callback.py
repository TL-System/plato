from __future__ import annotations

from types import SimpleNamespace

from plato.callbacks.server import LogProgressCallback
from plato.config import Config


def test_log_progress_callback_tolerates_missing_logged_items(temp_config, monkeypatch):
    written_rows: list[list[object]] = []

    monkeypatch.setattr(
        "plato.callbacks.server.csv_processor.initialize_csv",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "plato.callbacks.server.csv_processor.write_csv",
        lambda _path, row: written_rows.append(row),
    )

    Config.params["result_types"] = "round, accuracy, evaluation_ifeval_avg"

    callback = LogProgressCallback()
    server = SimpleNamespace(
        get_logged_items=lambda: {"round": 1, "accuracy": 0.5},
        updates=[],
    )

    callback.on_clients_processed(server)

    assert written_rows == [[1, 0.5, None]]
