from __future__ import annotations

import csv
import os
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


def test_log_progress_callback_expands_csv_for_new_lighteval_metrics(
    temp_config, tmp_path
):
    result_path = tmp_path / "results"
    result_path.mkdir()
    Config.params["result_path"] = str(result_path)
    Config.params["result_types"] = "round, accuracy, evaluation_ifeval_avg"

    callback = LogProgressCallback()
    server = SimpleNamespace(
        get_logged_items=lambda: {
            "round": 1,
            "accuracy": 0.5,
            "evaluation_ifeval_avg": 0.21875,
            "evaluation_arc_easy": 0.375,
            "evaluation_arc_challenge": 0.1875,
            "evaluation_ifeval_prompt_level_strict_acc": 0.21875,
        },
        updates=[],
    )

    callback.on_clients_processed(server)

    csv_path = os.path.join(result_path, f"{os.getpid()}.csv")
    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows[0] == [
        "round",
        "accuracy",
        "evaluation_ifeval_avg",
        "evaluation_arc_easy",
        "evaluation_arc_challenge",
        "evaluation_ifeval_prompt_level_strict_acc",
    ]
    assert rows[1] == [
        "1",
        "0.5",
        "0.21875",
        "0.375",
        "0.1875",
        "0.21875",
    ]
