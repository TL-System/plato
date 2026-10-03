"""Synchronous FedDyn cloud/history authority and atomic round bundles."""

from __future__ import annotations

import copy
import os
import random
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plato.callbacks.handler import CallbackHandler
from plato.config import Config
from plato.servers import fedavg
from plato.trainers.strategies.algorithms.feddyn_strategy import (
    effective_alpha,
    model_schema,
    positive_integer,
    same_state,
    settings_from_config,
    trainable_reference,
    validate_endpoint,
    validate_identity,
    validate_tensors,
)
from plato.utils.checkpoint_paths import checkpoint_name, checkpoint_path


class _Callbacks(CallbackHandler):
    def __init__(self, original, guard):
        self.callbacks = original.callbacks
        self.original, self.guard = original, guard

    def call_event(self, event, *args, **kwargs):
        self.original.call_event(event, *args, **kwargs)
        self.guard(event, *args, **kwargs)


class Server(fedavg.Server):
    """Commit mean(selected endpoints)+mean(all population histories).

    Counts affect only alpha_i in explicitly configured sample mode. Frozen
    parameters/static buffers remain equal to the dispatch baseline. This is
    a completed-round CPU replay contract, without in-flight or GPU replay.
    """

    def __init__(self, *args, **kwargs):
        self.settings: dict[str, Any] = settings_from_config()
        super().__init__(*args, **kwargs)
        self.run_id = uuid.uuid4().hex
        self.committed_round: int = 0
        self.histories: dict[int, dict[str, torch.Tensor]] = {}
        self.observed_counts: dict[int, int] = {}
        self.dispatches: dict[int, dict[str, Any]] = {}
        self.accepted_tokens: set[str] = set()
        self.schema = None
        self._pending: dict[str, Any] | None = None
        self._pending_seal: dict[str, Any] | None = None
        self._committed_snapshot: dict[str, Any] | None = None
        self._pending_resume_rng: dict[str, Any] | None = None

    def configure(self):
        super().configure()
        self._ensure_session()

    def _model(self):
        model = self.require_trainer().model
        if model is None:
            raise ValueError("FedDyn requires an ordinary torch model.")
        return model

    def _ensure_session(self):
        model = self._model()
        schema = model_schema(model)
        if self.schema is None:
            self.schema = schema
            q = trainable_reference(model)
            self.histories = {
                i: OrderedDict((k, torch.zeros_like(v).cpu()) for k, v in q.items())
                for i in range(1, self.settings["population"] + 1)
            }
            self._committed_snapshot = self._snapshot()
        elif schema != self.schema:
            raise ValueError(
                "FedDyn model schema changed; start an explicit fresh run."
            )

    def warm_start_model(self, weights):
        """Explicit model-only fresh run; old unversioned histories are ignored."""
        full = validate_tensors(weights, self._model().state_dict(), "warm-start model")
        self._model().load_state_dict(full)
        self.run_id = uuid.uuid4().hex
        self.committed_round = self.current_round = 0
        self.histories, self.observed_counts, self.dispatches = {}, {}, {}
        self.accepted_tokens = set()
        self.schema = None
        self._pending = self._pending_resume_rng = None
        self._ensure_session()

    def customize_server_response(self, server_response, client_id):
        self._ensure_session()
        i = positive_integer(client_id, "client ID")
        if (
            i > self.settings["population"]
            or self.current_round != self.committed_round + 1
        ):
            raise ValueError(
                "FedDyn requires the next synchronous committed-round dispatch."
            )
        if i not in self.selected_clients:
            raise ValueError("FedDyn dispatch must belong to the selected set.")
        if i not in self.dispatches:
            full = validate_tensors(
                self._model().state_dict(), self._model().state_dict(), "cloud"
            )
            count = (
                self.settings["sample_counts"][i - 1]
                if self.settings["sample_counts"] is not None
                else self.observed_counts.get(i)
            )
            alpha = (
                effective_alpha(
                    self.settings["base_alpha"],
                    self.settings["population"],
                    self.settings["sample_counts"],
                    i,
                )
                if self.settings["weighting"] == "sample"
                else self.settings["base_alpha"]
            )
            metadata = dict(
                version=1,
                run_id=self.run_id,
                round=self.current_round,
                client_id=i,
                dispatch_token=uuid.uuid4().hex,
                weighting=self.settings["weighting"],
                base_alpha=self.settings["base_alpha"],
                effective_alpha=alpha,
                expected_count_or_null=count,
                history=validate_tensors(
                    self.histories[i], trainable_reference(self._model()), "history"
                ),
            )
            self.dispatches[i] = dict(metadata=metadata, baseline=full)
        d = self.dispatches[i]["metadata"]
        if d["round"] != self.current_round:
            raise ValueError("FedDyn has a stale uncommitted dispatch.")
        return {
            **server_response,
            "feddyn": {
                k: d[k]
                for k in ("version", "run_id", "round", "client_id", "dispatch_token")
            },
        }

    def customize_server_payload(self, payload):
        record = self.dispatches.get(self.selected_client_id)
        if record is None:
            raise ValueError("FedDyn payload needs its assigned dispatch.")
        full = validate_tensors(payload, record["baseline"], "outbound cloud")
        if any(not torch.equal(v, record["baseline"][k]) for k, v in full.items()):
            raise ValueError("FedDyn cloud changed during round dispatch.")
        return [full, copy.deepcopy(record["metadata"])]

    def _validate_batch(self, updates, payloads):
        self._ensure_session()
        selected = self.selected_clients
        if self.current_round != self.committed_round + 1:
            raise ValueError("FedDyn requires the next uncommitted dispatch round.")
        if (
            not isinstance(selected, (list, tuple))
            or len(selected) != self.settings["per_round"]
        ):
            raise ValueError(
                "FedDyn requires its complete configured participating set."
            )
        for client_id in selected:
            if (
                positive_integer(client_id, "selected client ID")
                > self.settings["population"]
            ):
                raise ValueError("FedDyn selected client is outside the population.")
        if (
            not isinstance(selected, (list, tuple))
            or not selected
            or len(set(selected)) != len(selected)
            or set(selected) != set(self.dispatches)
            or len(updates) != len(selected)
            or len(payloads) != len(updates)
        ):
            raise ValueError(
                "FedDyn requires the entire unique dispatched participating set."
            )
        result, seen = [], set()
        fields = {
            "version",
            "run_id",
            "round",
            "client_id",
            "dispatch_token",
            "num_samples",
            "completed_steps",
        }
        for update, payload in zip(updates, payloads):
            if not isinstance(payload, (list, tuple)) or len(payload) != 2:
                raise ValueError("FedDyn requires full model and result metadata.")
            full, raw_metadata = payload
            m = raw_metadata
            if not isinstance(m, dict) or set(m) != fields:
                raise ValueError("FedDyn result metadata is incomplete or unknown.")
            i = positive_integer(m["client_id"], "client ID")
            report_id = positive_integer(update.report.client_id, "report client ID")
            if (
                i != report_id
                or (hasattr(update, "client_id") and update.client_id != i)
                or i in seen
                or i not in self.dispatches
            ):
                raise ValueError("FedDyn duplicate/unknown/mismatched logical client.")
            record = self.dispatches[i]
            expected = record["metadata"]
            if any(
                type(m[k]) is not type(expected[k]) or m[k] != expected[k]
                for k in ("version", "run_id", "round", "client_id", "dispatch_token")
            ):
                raise ValueError("FedDyn stale/mismatched dispatch identity.")
            if (
                m["round"] != self.committed_round + 1
                or m["dispatch_token"] in self.accepted_tokens
            ):
                raise ValueError("FedDyn dispatch was already consumed or is stale.")
            count = positive_integer(m["num_samples"], "result sample count")
            if (
                positive_integer(update.report.num_samples, "report sample count")
                != count
            ):
                raise ValueError("FedDyn report/result count mismatch.")
            if (
                expected["expected_count_or_null"] is not None
                and count != expected["expected_count_or_null"]
            ):
                raise ValueError("FedDyn realized count differs from fixed partition.")
            positive_integer(m["completed_steps"], "completed optimizer steps")
            y = validate_endpoint(self._model(), full, record["baseline"], self.schema)
            h = validate_tensors(
                self.histories[i],
                trainable_reference(self._model()),
                "committed history",
            )
            if any(not torch.equal(v, expected["history"][k]) for k, v in h.items()):
                raise ValueError("FedDyn committed history changed during dispatch.")
            seen.add(i)
            result.append((i, y, copy.deepcopy(m)))
        if seen != set(selected):
            raise ValueError("FedDyn is missing a dispatched client.")
        if any(
            not same_state(record["baseline"], self._model().state_dict())
            for record in self.dispatches.values()
        ):
            raise ValueError("FedDyn cloud changed during its fixed round dispatch.")
        return result

    def weights_received(self, weights_received):
        self._pending = None
        self._validate_batch(self.updates, weights_received)
        return copy.deepcopy(weights_received)

    def _should_prefer_weight_aggregation(self):
        return False

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        # Validate the actual callback-returned payloads and reports before any
        # history arithmetic; counts never enter either model mean.
        batch = self._validate_batch(updates, weights_received)
        histories = copy.deepcopy(self.histories)
        counts = dict(self.observed_counts)
        model = self._model()
        q = trainable_reference(model)
        for i, y, m in batch:
            x = self.dispatches[i]["baseline"]
            histories[i] = validate_tensors(
                {k: h + y[k] - x[k] for k, h in histories[i].items()}, q, "next history"
            )
            counts[i] = m["num_samples"]
        n = self.settings["population"]
        corrected = copy.deepcopy(self.dispatches[batch[0][0]]["baseline"])
        for k in q:
            corrected[k] = (
                sum(y[k] for _, y, _ in batch) / len(batch)
                + sum(histories[i][k] for i in range(1, n + 1)) / n
            )
        corrected = validate_endpoint(
            model, corrected, self.dispatches[batch[0][0]]["baseline"], self.schema
        )
        self._pending = dict(
            histories=histories,
            counts=counts,
            model=corrected,
            tokens={m["dispatch_token"] for _, _, m in batch},
            round=self.current_round,
        )
        self._pending_seal = copy.deepcopy(self._pending)
        return copy.deepcopy(corrected)

    def weights_aggregated(self, updates):
        if self._pending is None or not same_state(self._pending, self._pending_seal):
            raise ValueError("FedDyn has no validated pending batch.")
        pending = self._pending
        full = validate_tensors(
            self._model().state_dict(), pending["model"], "aggregated cloud"
        )
        if any(not torch.equal(v, pending["model"][k]) for k, v in full.items()):
            raise ValueError("FedDyn loaded model differs from staged cloud.")
        self.histories = copy.deepcopy(pending["histories"])
        self.observed_counts = dict(pending["counts"])
        self.committed_round = pending["round"]
        self.accepted_tokens |= pending["tokens"]

    async def _process_reports(self):
        """Bound only model/history/count/token mutation through callbacks."""
        self._pending = None
        self._pending_seal = None
        self._ensure_session()
        model = self._model()
        before_model = copy.deepcopy(model.state_dict())
        before: tuple[
            dict[int, dict[str, torch.Tensor]],
            dict[int, int],
            int,
            set[str],
            dict[str, Any] | None,
        ] = copy.deepcopy(
            (
                self.histories,
                self.observed_counts,
                self.committed_round,
                self.accepted_tokens,
                self._committed_snapshot,
            )
        )
        before_dispatches = copy.deepcopy(self.dispatches)
        before_run, before_schema = self.run_id, copy.deepcopy(self.schema)
        before_settings = copy.deepcopy(self.settings)
        before_round, before_selected = self.current_round, list(self.selected_clients)
        handler = self.callback_handler
        committed = False

        def guard(event, *args, **kwargs):
            nonlocal committed
            if event == "on_weights_received":
                if (
                    set(self.histories) != set(before[0])
                    or self.observed_counts != before[1]
                    or self.committed_round != before[2]
                    or self.accepted_tokens != before[3]
                    or self.run_id != before_run
                    or self.schema != before_schema
                    or self.settings != before_settings
                    or self.current_round != before_round
                    or self.selected_clients != before_selected
                    or set(self.dispatches) != set(before_dispatches)
                ):
                    raise ValueError(
                        "FedDyn receive callback changed committed metadata."
                    )
                for i, expected in before[0].items():
                    actual = validate_tensors(
                        self.histories[i], expected, "callback committed history"
                    )
                    if any(not torch.equal(v, expected[k]) for k, v in actual.items()):
                        raise ValueError(
                            "FedDyn receive callback changed committed history."
                        )
                for i, old in before_dispatches.items():
                    current = self.dispatches[i]
                    if current["metadata"].keys() != old["metadata"].keys() or any(
                        current["metadata"][k] != old["metadata"][k]
                        for k in old["metadata"]
                        if k != "history"
                    ):
                        raise ValueError("FedDyn callback changed dispatch identity.")
                    for key in ("baseline",):
                        actual = validate_tensors(
                            current[key], old[key], "callback dispatch"
                        )
                        if any(
                            not torch.equal(v, old[key][k]) for k, v in actual.items()
                        ):
                            raise ValueError(
                                "FedDyn callback changed dispatch baseline."
                            )
                    actual = validate_tensors(
                        current["metadata"]["history"],
                        old["metadata"]["history"],
                        "callback dispatch history",
                    )
                    if any(
                        not torch.equal(v, old["metadata"]["history"][k])
                        for k, v in actual.items()
                    ):
                        raise ValueError("FedDyn callback changed dispatch history.")
                self._validate_batch(self.updates, args[1])
            if event == "on_weights_aggregated":
                pending = self._pending_seal
                if pending is None or not same_state(self._pending, pending):
                    raise ValueError("FedDyn aggregation callback discarded staging.")
                if (
                    self.run_id != before_run
                    or not same_state(self.schema, before_schema)
                    or not same_state(self.settings, before_settings)
                    or self.current_round != before_round
                    or self.selected_clients != before_selected
                    or set(self.histories) != set(pending["histories"])
                ):
                    raise ValueError(
                        "FedDyn aggregation callback changed round authority."
                    )
                full = validate_tensors(
                    model.state_dict(), pending["model"], "callback cloud"
                )
                if any(
                    not torch.equal(v, pending["model"][k]) for k, v in full.items()
                ):
                    raise ValueError(
                        "FedDyn aggregation callback changed staged model."
                    )
                for i, expected in pending["histories"].items():
                    actual = validate_tensors(
                        self.histories[i], expected, "callback history"
                    )
                    if any(not torch.equal(v, expected[k]) for k, v in actual.items()):
                        raise ValueError("FedDyn callback changed staged history.")
                if (
                    self.observed_counts != pending["counts"]
                    or self.committed_round != pending["round"]
                    or self.accepted_tokens != before[3] | pending["tokens"]
                ):
                    raise ValueError("FedDyn callback changed committed metadata.")
                self._committed_snapshot = self._snapshot()
                self.dispatches = {}
                committed = True

        self.callback_handler = _Callbacks(handler, guard)
        try:
            await super()._process_reports()
        except BaseException:
            if not committed:
                with torch.no_grad():
                    for name, tensor in model.state_dict().items():
                        tensor.copy_(before_model[name])
                (
                    self.histories,
                    self.observed_counts,
                    self.committed_round,
                    self.accepted_tokens,
                    self._committed_snapshot,
                ) = before
                self.dispatches = before_dispatches
                self.run_id, self.schema = before_run, before_schema
                self.settings = before_settings
                self.current_round, self.selected_clients = (
                    before_round,
                    before_selected,
                )
            raise
        finally:
            self.callback_handler = handler
            self._pending = None
            self._pending_seal = None

    def _rng_snapshot(self):
        numpy_state = np.random.get_state()
        return dict(
            selection=copy.deepcopy(self.prng_state),
            context_selection=copy.deepcopy(
                self.context.state.get("prng_state", self.prng_state)
            ),
            python=random.getstate(),
            numpy=[
                numpy_state[0],
                torch.from_numpy(numpy_state[1].copy()),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            ],
            torch=torch.get_rng_state().clone(),
        )

    def _snapshot(self):
        return dict(
            version=1,
            run_id=self.run_id,
            committed_round=self.committed_round,
            config=copy.deepcopy(self.settings),
            schema=copy.deepcopy(self.schema),
            model=copy.deepcopy(self._model().state_dict()),
            histories=copy.deepcopy(self.histories),
            counts=dict(self.observed_counts),
            accepted_tokens=sorted(self.accepted_tokens),
            rng=self._rng_snapshot(),
        )

    def checkpoint_bundle_path(self):
        name = getattr(Config().trainer, "model_name", "custom")
        return checkpoint_path(
            Config().params["checkpoint_path"],
            checkpoint_name("feddyn", name, suffix=".pth"),
        )

    def save_to_checkpoint(self):
        """Atomically publish the coherent last committed CPU bundle."""
        self._ensure_session()
        snapshot = copy.deepcopy(self._committed_snapshot)
        self._validate_bundle(snapshot)
        path = Path(self.checkpoint_bundle_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".feddyn-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(snapshot, stream)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _validate_rng(rng):
        if not isinstance(rng, dict) or set(rng) != {
            "selection",
            "context_selection",
            "python",
            "numpy",
            "torch",
        }:
            raise ValueError("FedDyn checkpoint RNG fields are incomplete.")
        if rng["selection"] != rng["context_selection"]:
            raise ValueError("FedDyn duplicated selection states disagree.")
        try:
            random.Random().setstate(rng["selection"])
            random.Random().setstate(rng["python"])
            state = rng["numpy"]
            if (
                not isinstance(state, (list, tuple))
                or len(state) != 5
                or not isinstance(state[1], torch.Tensor)
                or state[1].dtype != torch.uint32
                or state[1].shape != (624,)
                or state[0] != "MT19937"
                or type(state[2]) is not int
                or not 0 <= state[2] <= 624
                or type(state[3]) is not int
                or state[3] not in (0, 1)
                or type(state[4]) is not float
                or not np.isfinite(state[4])
            ):
                raise ValueError("Malformed NumPy state.")
            np.random.RandomState().set_state(
                (state[0], state[1].cpu().numpy().copy(), state[2], state[3], state[4])
            )
            torch.Generator(device="cpu").set_state(rng["torch"])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("FedDyn checkpoint has invalid CPU RNG state.") from exc
        return copy.deepcopy(rng)

    def _validate_bundle(self, bundle):
        keys = {
            "version",
            "run_id",
            "committed_round",
            "config",
            "schema",
            "model",
            "histories",
            "counts",
            "accepted_tokens",
            "rng",
        }
        if not isinstance(bundle, dict) or set(bundle) != keys:
            raise ValueError(
                "FedDyn requires a complete versioned history bundle; old model-only checkpoints require an explicit fresh warm start."
            )
        if type(bundle["version"]) is not int or bundle["version"] != 1:
            raise ValueError("FedDyn checkpoint version must be 1.")
        if not same_state(bundle["config"], self.settings) or not same_state(
            bundle["schema"], model_schema(self._model())
        ):
            raise ValueError(
                "FedDyn checkpoint configuration/schema changed; use an explicit fresh run."
            )
        validate_identity(bundle["run_id"], "checkpoint run ID")
        round_id = bundle["committed_round"]
        if type(round_id) is not int or round_id < 0:
            raise ValueError("FedDyn committed round must be a nonnegative integer.")
        full = validate_tensors(
            bundle["model"], self._model().state_dict(), "checkpoint model"
        )
        n = self.settings["population"]
        histories = bundle["histories"]
        if (
            not isinstance(histories, dict)
            or any(type(i) is not int for i in histories)
            or set(histories) != set(range(1, n + 1))
        ):
            raise ValueError("FedDyn checkpoint must contain every population history.")
        h = {
            i: validate_tensors(
                histories[i], trainable_reference(self._model()), "checkpoint history"
            )
            for i in range(1, n + 1)
        }
        counts = bundle["counts"]
        if not isinstance(counts, dict):
            raise ValueError("FedDyn checkpoint counts are malformed.")
        for i, count in counts.items():
            if positive_integer(i, "count client ID") > n:
                raise ValueError("FedDyn checkpoint count client is unknown.")
            count = positive_integer(count, "checkpoint sample count")
            if (
                self.settings["sample_counts"] is not None
                and count != self.settings["sample_counts"][i - 1]
            ):
                raise ValueError(
                    "FedDyn checkpoint counts differ from population vector."
                )
        tokens = bundle["accepted_tokens"]
        if not isinstance(tokens, list) or len(set(tokens)) != len(tokens):
            raise ValueError("FedDyn checkpoint token ledger is malformed.")
        for token in tokens:
            validate_identity(token, "accepted token")
        if (
            len(tokens) != round_id * self.settings["per_round"]
            or (round_id == 0 and counts)
            or (round_id > 0 and not counts)
        ):
            raise ValueError(
                "FedDyn checkpoint round/count/token ledger is inconsistent."
            )
        rng = self._validate_rng(bundle["rng"])
        return {**copy.deepcopy(bundle), "model": full, "histories": h, "rng": rng}

    def _resume_from_checkpoint(self):
        path = self.checkpoint_bundle_path()
        if not os.path.isfile(path):
            raise ValueError(
                "FedDyn resume needs its complete versioned bundle; inspect legacy history or explicitly warm-start a new run."
            )
        bundle = self._validate_bundle(
            torch.load(path, weights_only=True, map_location="cpu")
        )
        # Validation above touches only private generators/owned copies. Install
        # known validated tensors without a partial custom loader mutation.
        with torch.no_grad():
            for k, tensor in self._model().state_dict().items():
                tensor.copy_(bundle["model"][k])
        self.schema = bundle["schema"]
        self.histories, self.observed_counts = bundle["histories"], bundle["counts"]
        self.run_id, self.committed_round = bundle["run_id"], bundle["committed_round"]
        self.current_round = self.context.current_round = self.committed_round
        self.resumed_session = True
        self.dispatches = {}
        self.accepted_tokens = set(bundle["accepted_tokens"])
        self._committed_snapshot = copy.deepcopy(bundle)
        self._pending_resume_rng = dict(
            run_id=self.run_id,
            round=self.committed_round,
            schema=copy.deepcopy(self.schema),
            rng=copy.deepcopy(bundle["rng"]),
        )

    def start(self, *args, **kwargs):
        """Install every promised CPU RNG once, before inherited registration."""
        pending = self._pending_resume_rng
        if pending is not None:
            if (
                pending["run_id"] != self.run_id
                or pending["round"] != self.committed_round
                or pending["schema"] != model_schema(self._model())
            ):
                raise ValueError("FedDyn pending resume identity/schema changed.")
            rng = self._validate_rng(pending["rng"])
            previous = self._rng_snapshot()
            try:
                self.prng_state = copy.deepcopy(rng["selection"])
                self.context.state["prng_state"] = copy.deepcopy(
                    rng["context_selection"]
                )
                random.setstate(rng["python"])
                ns = rng["numpy"]
                np.random.set_state((ns[0], ns[1].numpy().copy(), ns[2], ns[3], ns[4]))
                torch.set_rng_state(rng["torch"])
            except BaseException:
                self.prng_state = previous["selection"]
                self.context.state["prng_state"] = previous["context_selection"]
                random.setstate(previous["python"])
                ns = previous["numpy"]
                np.random.set_state((ns[0], ns[1].numpy().copy(), ns[2], ns[3], ns[4]))
                torch.set_rng_state(previous["torch"])
                raise
            self._pending_resume_rng = None
        return super().start(*args, **kwargs)
