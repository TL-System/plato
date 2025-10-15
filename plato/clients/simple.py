"""
A basic federated learning client who sends weight updates to the server.
"""

from plato.clients import base
from plato.clients.strategies import (
    DefaultCommunicationStrategy,
    DefaultLifecycleStrategy,
    DefaultPayloadStrategy,
    DefaultReportingStrategy,
    DefaultTrainingStrategy,
)


class Client(base.Client):
    """A basic federated learning client who composes default strategies."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        super().__init__(callbacks=callbacks)

        # Preserve legacy attributes for backward compatibility
        self.custom_model = model
        self.custom_datasource = datasource
        self.custom_algorithm = algorithm
        self.custom_trainer = trainer
        self.trainer_callbacks = trainer_callbacks

        self.model = None
        self.datasource = None
        self.algorithm = None
        self.trainer = None
        self.trainset = None
        self.testset = None
        self.sampler = None
        self.testset_sampler = None
        self._report = None

        # Mirror configuration metadata into the shared context
        self._context.custom_model = model
        self._context.custom_datasource = datasource
        self._context.custom_algorithm = algorithm
        self._context.custom_trainer = trainer
        self._context.trainer_callbacks = trainer_callbacks
        self._context.report_customizer = self.customize_report

        # Default strategy stack reproducing the legacy client behaviour
        self._configure_composable(
            lifecycle_strategy=DefaultLifecycleStrategy(),
            payload_strategy=DefaultPayloadStrategy(),
            training_strategy=DefaultTrainingStrategy(),
            reporting_strategy=DefaultReportingStrategy(),
            communication_strategy=DefaultCommunicationStrategy(),
        )

    def configure(self) -> None:
        """Prepare this client for training using lifecycle strategies."""
        self._context.custom_model = self.custom_model
        self._context.custom_trainer = self.custom_trainer
        self._context.custom_algorithm = self.custom_algorithm
        self._context.trainer_callbacks = self.trainer_callbacks
        self._context.report_customizer = self.customize_report

        self.lifecycle_strategy.configure(self._context)
        self._sync_from_context(
            (
                "model",
                "trainer",
                "algorithm",
                "outbound_processor",
                "inbound_processor",
                "sampler",
                "testset_sampler",
            )
        )

    def _load_data(self) -> None:
        """Generate data sources using the configured lifecycle strategy."""
        self._context.custom_datasource = self.custom_datasource
        self.lifecycle_strategy.load_data(self._context)
        self._sync_from_context(("datasource",))

    def _allocate_data(self) -> None:
        """Allocate train/test datasets via the lifecycle strategy."""
        self.lifecycle_strategy.allocate_data(self._context)
        self._sync_from_context(("trainset", "testset"))

    def _load_payload(self, server_payload) -> None:
        """Load inbound payload via the training strategy."""
        self.training_strategy.load_payload(self._context, server_payload)

    async def _train(self):
        """Run local training using the configured training strategy."""
        report, payload = await self.training_strategy.train(self._context)
        report = self.reporting_strategy.build_report(self._context, report)
        self._context.latest_report = report
        self._report = report
        return report, payload

    async def _obtain_model_at_time(self, client_id, requested_time):
        """Delegate asynchronous model retrieval to the reporting strategy."""
        return await self.reporting_strategy.obtain_model_at_time(
            self._context, client_id, requested_time
        )

    def customize_report(self, report):
        """Customizes the report with any additional information."""
        return report
