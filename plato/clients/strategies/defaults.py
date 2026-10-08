"""
Default client strategy implementations matching the legacy behaviour.

These strategies provide a drop-in equivalent of the current
`plato.clients.simple.Client` pipeline. A future composable client will wire
them together to offer parity before incremental refactoring of specialised
clients.
"""

from __future__ import annotations

import logging
import pickle
import re
import sys
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from plato.algorithms import registry as algorithms_registry
from plato.clients.strategies.base import (
    ClientContext,
    CommunicationStrategy,
    LifecycleStrategy,
    PayloadStrategy,
    ReportingStrategy,
    TrainingStrategy,
)
from plato.clients.transport import (
    InboundTransfer,
    TransportLimits,
    receive_s3_payload,
)
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry
from plato.utils import fonts
from plato.utils.checkpoint_paths import checkpoint_name, checkpoint_path

LOGGER = logging.getLogger(__name__)


class DefaultLifecycleStrategy(LifecycleStrategy):
    """Legacy lifecycle implementation copied from `simple.Client`."""

    def process_server_response(
        self, context: ClientContext, server_response: dict[str, Any]
    ) -> None:
        """
        Default implementation mirrors legacy behaviour (no-op).

        Custom clients that previously overrode `process_server_response`
        will be migrated to bespoke strategies in later steps.
        """

    def load_data(self, context: ClientContext) -> None:
        """Generate and cache the datasource for the client."""
        custom_datasource = getattr(context, "custom_datasource", None)

        should_reload = bool(
            getattr(context, "datasource", None) is None
            or (hasattr(Config().data, "reload_data") and Config().data.reload_data)
        )

        if not should_reload:
            return

        LOGGER.info("[%s] Loading its data source...", context)

        if custom_datasource is None:
            context.datasource = datasources_registry.get(client_id=context.client_id)
        else:
            context.datasource = custom_datasource()

        LOGGER.info(
            "[%s] Dataset size: %s",
            context,
            context.datasource.num_train_examples(),
        )

    def configure(self, context: ClientContext) -> None:
        """Instantiate trainer, algorithm, and payload processors."""
        custom_model = getattr(context, "custom_model", None)
        custom_trainer = getattr(context, "custom_trainer", None)
        custom_algorithm = getattr(context, "custom_algorithm", None)

        if context.model is None and custom_model is not None:
            context.model = custom_model

        if context.trainer is None:
            if custom_trainer is None:
                context.trainer = trainers_registry.get(
                    model=context.model, callbacks=context.trainer_callbacks
                )
            else:
                context.trainer = custom_trainer(
                    model=context.model, callbacks=context.trainer_callbacks
                )

        if context.trainer is not None:
            context.trainer.set_client_id(context.client_id)

        if context.algorithm is None:
            if custom_algorithm is None:
                context.algorithm = algorithms_registry.get(trainer=context.trainer)
            else:
                context.algorithm = custom_algorithm(trainer=context.trainer)

        if context.algorithm is not None:
            context.algorithm.set_client_id(context.client_id)

        processor_kwargs = getattr(context, "processor_kwargs", None)
        processors = processor_registry.get(
            "Client",
            processor_kwargs=processor_kwargs,
            client_id=context.client_id,
            trainer=context.trainer,
        )
        context.outbound_processor, context.inbound_processor = processors

        if context.datasource is None:
            return

        context.sampler = samplers_registry.get(context.datasource, context.client_id)

        if (
            hasattr(Config().clients, "do_test")
            and Config().clients.do_test
            and hasattr(Config().data, "testset_sampler")
        ):
            context.testset_sampler = samplers_registry.get(
                context.datasource, context.client_id, testing=True
            )

    def allocate_data(self, context: ClientContext) -> None:
        """Allocate train and optional test data to the client."""
        if context.datasource is None:
            return

        context.trainset = context.datasource.get_train_set()

        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            context.testset = context.datasource.get_test_set()


class DefaultPayloadStrategy(PayloadStrategy):
    """Default payload processing, mirroring `Client._handle_payload`."""

    def reset_payload(self, context: ClientContext) -> None:
        self._clear_transfer(context)
        super().reset_payload(context)
        if not context.comm_simulation:
            transfer = InboundTransfer(
                TransportLimits.from_config(), lambda: self._clear_transfer(context)
            )
            context.state["inbound_transfer"] = transfer
            context.chunks = transfer.chunks

    @staticmethod
    def _clear_transfer(context: ClientContext) -> None:
        transfer = context.state.pop("inbound_transfer", None)
        if transfer is not None:
            transfer.close()
        context.chunks.clear()
        context.server_payload = None
        if context.owner is not None:
            context.owner.chunks = context.chunks
            context.owner.server_payload = None

    def teardown(self, context: ClientContext) -> None:
        self._clear_transfer(context)

    @staticmethod
    def _require_transfer(context: ClientContext) -> InboundTransfer:
        transfer = context.state.get("inbound_transfer")
        if transfer is None:
            raise ValueError("No active server payload transfer.")
        transfer.check_active()
        return transfer

    async def accumulate_chunk(self, context: ClientContext, chunk: bytes) -> None:
        try:
            self._require_transfer(context).append(chunk)
        except Exception:
            self._clear_transfer(context)
            raise

    def inbound_received(self, context: ClientContext) -> None:
        """Invoke legacy hook on the owning client."""
        owner = context.owner
        inbound_processor = context.inbound_processor
        if owner is not None:
            owner.inbound_received(inbound_processor)

    def outbound_ready(
        self,
        context: ClientContext,
        report: Any,
        outbound_payload: Any,
    ) -> None:
        """Invoke legacy outbound hook on the owning client."""
        owner = context.owner
        outbound_processor = context.outbound_processor
        if owner is not None:
            owner.outbound_ready(report, outbound_processor)

    async def commit_chunk_group(
        self,
        context: ClientContext,
        client_id: int,
    ) -> None:
        """Commit buffered chunks similarly to `_payload_arrived`."""
        try:
            if client_id != context.client_id:
                raise ValueError("Chunk group client ID does not match the active client.")
            context.server_payload = self._require_transfer(context).commit()
        except Exception:
            self._clear_transfer(context)
            raise

    async def finalise_inbound_payload(
        self,
        context: ClientContext,
        client_id: int,
        *,
        s3_key: str | None = None,
    ) -> Any:
        """Reconstruct inbound payload and log payload statistics."""
        try:
            if client_id != context.client_id:
                raise ValueError(
                    "Payload completion client ID does not match the active client."
                )
            transfer = self._require_transfer(context)
            if s3_key is not None:
                if context.s3_client is None:
                    raise RuntimeError("S3 client not initialised for payload.")
                if (
                    not isinstance(s3_key, str)
                    or re.fullmatch(
                        rf"server_payload_\d+_{context.current_round}", s3_key
                    ) is None
                ):
                    raise ValueError("S3 payload key does not match the current round.")
                if transfer.chunk_count or transfer.part_count:
                    raise ValueError("S3 payload cannot replace an active socket payload.")
                payload = await receive_s3_payload(
                    context.s3_client, s3_key, transfer, transfer.reserve
                )
            else:
                payload = transfer.finish()
            payload_size = transfer.byte_count
        except BaseException:
            self._clear_transfer(context)
            raise
        self._clear_transfer(context)
        context.server_payload = payload

        LOGGER.info(
            "[Client #%d] Received %.2f MB of payload data from the server.",
            client_id,
            payload_size / 1024**2,
        )

        return context.server_payload

    async def handle_server_payload(
        self,
        context: ClientContext,
        server_payload: Any,
        *,
        training: TrainingStrategy,
        reporting: ReportingStrategy,
        communication: CommunicationStrategy,
    ) -> None:
        """Run inbound processing, training, reporting, and outbound send."""
        callbacks = context.callback_handler
        owner = context.owner or context
        inbound_processor = context.inbound_processor
        outbound_processor = context.outbound_processor

        self.inbound_received(context)

        if callbacks is not None:
            callbacks.call_event("on_inbound_received", owner, inbound_processor)

        tic = time.perf_counter()
        if inbound_processor is not None:
            processed_inbound = inbound_processor.process(server_payload)
        else:
            processed_inbound = server_payload
        context.processing_time = time.perf_counter() - tic

        training.load_payload(context, processed_inbound)
        report, outbound_payload = await training.train(context)

        if callbacks is not None:
            callbacks.call_event("on_inbound_processed", owner, processed_inbound)

        report = reporting.build_report(context, report)

        self.outbound_ready(context, report, outbound_payload)

        if callbacks is not None:
            callbacks.call_event("on_outbound_ready", owner, report, outbound_processor)

        tic = time.perf_counter()
        if outbound_processor is not None:
            processed_outbound = outbound_processor.process(outbound_payload)
        else:
            processed_outbound = outbound_payload

        context.processing_time += time.perf_counter() - tic

        try:
            setattr(report, "processing_time", context.processing_time)
        except AttributeError:
            LOGGER.debug(
                "Unable to attach processing_time attribute to report %s.",
                report,
            )

        await communication.send_report_and_payload(context, report, processed_outbound)


class DefaultTrainingStrategy(TrainingStrategy):
    """Training behaviour identical to `simple.Client`."""

    def load_payload(self, context: ClientContext, server_payload: Any) -> None:
        if context.algorithm is None:
            raise RuntimeError("Algorithm is required before loading payload.")
        context.algorithm.load_weights(server_payload)

    async def train(self, context: ClientContext) -> tuple[Any, Any]:
        LOGGER.info(
            fonts.colourize(
                f"[{context}] Started training in communication "
                f"round #{context.current_round}."
            )
        )

        training_time = 0

        training_error = None

        try:
            if hasattr(context.trainer, "current_round"):
                context.trainer.current_round = context.current_round
            training_time = context.trainer.train(context.trainset, context.sampler)
        except ValueError as err:
            LOGGER.info(
                fonts.colourize(f"[{context}] Error occurred during training: {err}")
            )
            if context.sio is not None:
                await context.sio.disconnect()
            training_error = err

        if training_error is not None:
            context.state["training_error"] = training_error

        weights = context.algorithm.extract_weights()

        should_test = hasattr(Config().clients, "do_test") and Config().clients.do_test
        interval = getattr(Config().clients, "test_interval", None)
        run_test = should_test and (
            interval is None or context.current_round % interval == 0
        )

        accuracy = 0
        if run_test:
            accuracy = context.trainer.test(context.testset, context.testset_sampler)

            if accuracy == -1:
                LOGGER.info(
                    fonts.colourize(
                        f"[{context}] Accuracy is -1 when testing. Disconnecting."
                    )
                )
                if context.sio is not None:
                    await context.sio.disconnect()

            metric_name = getattr(
                getattr(context.trainer, "testing_strategy", None),
                "metric_name",
                "accuracy",
            )
            if metric_name == "mse":
                LOGGER.info("[%s] Test MSE: %.6f", context, accuracy)
            elif hasattr(Config().trainer, "target_perplexity"):
                LOGGER.info("[%s] Test perplexity: %.2f", context, accuracy)
            else:
                LOGGER.info("[%s] Test accuracy: %.2f%%", context, 100 * accuracy)

        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_times = Config().client_sleep_times
            if sleep_times is None:
                sleep_times = Config.simulate_client_speed()
            index = max(context.client_id - 1, 0)
            sleep_seconds = float(sleep_times[index])
            avg_training_time = Config().clients.avg_training_time
            training_time = (
                avg_training_time + sleep_seconds
            ) * Config().trainer.epochs

        num_samples = 0
        sampler = context.sampler
        if sampler is not None:
            if hasattr(sampler, "num_samples"):
                try:
                    num_samples = sampler.num_samples()
                except TypeError:
                    num_samples = 0
            if num_samples == 0 and hasattr(sampler, "__len__"):
                try:
                    num_samples = len(sampler)
                except TypeError:
                    num_samples = 0

        if num_samples == 0:
            num_samples = context.state.get("num_samples", 0)

        if num_samples == 0:
            trainset = getattr(context, "trainset", None)
            if trainset is not None and hasattr(trainset, "__len__"):
                try:
                    num_samples = len(trainset)
                except TypeError:
                    num_samples = 0

        # Extract train_loss from trainer's run_history if available
        train_loss = None
        if (
            context.trainer is not None
            and hasattr(context.trainer, "run_history")
            and context.trainer.run_history is not None
        ):
            try:
                train_loss = context.trainer.run_history.get_latest_metric("train_loss")
            except (AttributeError, KeyError, IndexError, ValueError):
                train_loss = None

        report = SimpleNamespace(
            client_id=context.client_id,
            num_samples=num_samples,
            accuracy=accuracy,
            training_time=training_time,
            comm_time=time.time(),
            update_response=False,
            train_loss=train_loss,
        )

        return report, weights


class DefaultReportingStrategy(ReportingStrategy):
    """Reporting hooks compatible with the legacy client."""

    def build_report(self, context: ClientContext, report: Any) -> Any:
        customizer = context.report_customizer
        final_report = report
        if callable(customizer):
            final_report = customizer(report)

        context.latest_report = final_report
        return final_report

    async def obtain_model_at_time(
        self, context: ClientContext, client_id: int, requested_time: float
    ) -> tuple[Any, Any]:
        if context.trainer is None or context.algorithm is None:
            raise RuntimeError("Trainer and algorithm are required.")

        model = context.trainer.obtain_model_at_time(client_id, requested_time)
        weights = context.algorithm.extract_weights(model)

        if context.latest_report is None:
            raise RuntimeError("Latest report is not available for async update.")

        context.latest_report.comm_time = time.time()
        context.latest_report.client_id = client_id
        context.latest_report.update_response = True

        return context.latest_report, weights


class DefaultCommunicationStrategy(CommunicationStrategy):
    """Communication behaviour lifted from the legacy client implementation."""

    def __init__(self, chunk_size: int = 1024**2) -> None:
        self.chunk_size = chunk_size

    async def send_report_and_payload(
        self, context: ClientContext, report: Any, payload: Any
    ) -> None:
        # Urgent responses can refer to an earlier logical client on this worker.
        context.state["outbound_client_id"] = getattr(
            report, "client_id", context.client_id
        )
        try:
            await super().send_report_and_payload(context, report, payload)
        finally:
            context.state.pop("outbound_client_id", None)

    async def send_report(self, context: ClientContext, report: Any) -> None:
        if context.sio is None:
            raise RuntimeError("Socket client not initialised.")

        await context.sio.emit(
            "client_report",
            {
                "id": context.state.get("outbound_client_id", context.client_id),
                "report": pickle.dumps(report),
            },
        )

    async def send_payload(self, context: ClientContext, payload: Any) -> None:
        client_id = context.state.get("outbound_client_id", context.client_id)
        if context.comm_simulation:
            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            checkpoint_root = Config().params["checkpoint_path"]
            payload_filename = checkpoint_path(
                checkpoint_root,
                checkpoint_name(model_name, "client", client_id, suffix=".pkl"),
            )

            with open(payload_filename, "wb") as payload_file:
                pickle.dump(payload, payload_file)

            data_size = sys.getsizeof(pickle.dumps(payload))

            LOGGER.info(
                "[%s] Sent %.2f MB of payload data to the server (simulated).",
                context,
                data_size / 1024**2,
            )
            return

        if context.sio is None:
            raise RuntimeError("Socket client not initialised.")

        metadata: dict[str, Any] = {"id": client_id}

        if context.s3_client is not None:
            unique_key = uuid.uuid4().hex[:6].upper()
            s3_key = f"client_payload_{client_id}_{unique_key}"
            context.s3_client.send_to_s3(s3_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
            metadata["s3_key"] = s3_key
        else:
            if isinstance(payload, list):
                data_size = 0
                for item in payload:
                    raw = pickle.dumps(item)
                    await self.send_in_chunks(context, raw)
                    data_size += sys.getsizeof(raw)
            else:
                raw = pickle.dumps(payload)
                await self.send_in_chunks(context, raw)
                data_size = sys.getsizeof(raw)

        await context.sio.emit("client_payload_done", metadata)

        LOGGER.info(
            "[%s] Sent %.2f MB of payload data to the server.",
            context,
            data_size / 1024**2,
        )

    async def send_in_chunks(self, context: ClientContext, data: bytes) -> None:
        if context.sio is None:
            raise RuntimeError("Socket client not initialised.")

        for start in range(0, len(data), self.chunk_size):
            chunk = data[start : start + self.chunk_size]
            await context.sio.emit("chunk", {"data": chunk})

        await context.sio.emit(
            "client_payload",
            {"id": context.state.get("outbound_client_id", context.client_id)},
        )
