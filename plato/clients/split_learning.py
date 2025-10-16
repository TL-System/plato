"""
A federated learning client using split learning.

Reference:

Vepakomma, et al., "Split Learning for Health: Distributed Deep Learning without Sharing
Raw Patient Data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf

Chopra, Ayush, et al. "AdaSplit: Adaptive Trade-offs for Resource-constrained Distributed
Deep Learning." arXiv preprint arXiv:2112.01637 (2021).

https://arxiv.org/pdf/2112.01637.pdf
"""

from plato.clients import simple
from plato.clients.strategies import SplitLearningTrainingStrategy
from plato.config import Config


class Client(simple.Client):
    """A split learning client."""

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        assert not Config().clients.do_test

        split_state = self._context.state.setdefault(
            "split_learning",
            {},
        )
        iterations = getattr(Config().clients, "iteration", 1)
        split_state.setdefault("iterations", iterations)
        split_state.setdefault("iter_left", iterations)
        split_state.setdefault("contexts", {})
        split_state.setdefault("original_weights", None)
        split_state.setdefault("static_sampler", None)

        self._configure_composable(
            lifecycle_strategy=self.lifecycle_strategy,
            payload_strategy=self.payload_strategy,
            training_strategy=SplitLearningTrainingStrategy(),
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )

    @property
    def iterations(self):
        """Total number of local iterations for split learning."""
        return self._context.state["split_learning"]["iterations"]

    @iterations.setter
    def iterations(self, value):
        self._context.state["split_learning"]["iterations"] = value

    @property
    def iter_left(self):
        """Remaining iterations before sending weights."""
        return self._context.state["split_learning"]["iter_left"]

    @iter_left.setter
    def iter_left(self, value):
        self._context.state["split_learning"]["iter_left"] = value

    @property
    def contexts(self):
        """Stored contexts for split learning sessions."""
        return self._context.state["split_learning"]["contexts"]

    @contexts.setter
    def contexts(self, value):
        self._context.state["split_learning"]["contexts"] = value

    @property
    def original_weights(self):
        """Cached original model weights."""
        return self._context.state["split_learning"]["original_weights"]

    @original_weights.setter
    def original_weights(self, value):
        self._context.state["split_learning"]["original_weights"] = value

    @property
    def static_sampler(self):
        """Sampler snapshot used during split learning rounds."""
        return self._context.state["split_learning"]["static_sampler"]

    @static_sampler.setter
    def static_sampler(self, value):
        self._context.state["split_learning"]["static_sampler"] = value
