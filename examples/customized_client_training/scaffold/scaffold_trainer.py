"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import SCAFFOLDUpdateStrategy


class Trainer(ComposableTrainer):
    """
    The federated learning trainer for the SCAFFOLD client.

    This trainer uses the composition-based design with SCAFFOLD update strategy.
    The SCAFFOLD algorithm uses control variates to correct for client drift.

    The server must provide the server control variate via:
        context.state['server_control_variate'] = server_control_variate

    After training, the client returns the control variate delta via:
        payload = trainer.model_update_strategy.get_update_payload(trainer.context)
        delta = payload['control_variate_delta']
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the SCAFFOLD trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=SCAFFOLDUpdateStrategy(),
        )

        # Store additional_data for server control variate
        # This maintains compatibility with the server-side code
        self.additional_data = None

    @property
    def client_control_variate_delta(self):
        """
        Property to expose the client control variate delta.

        This provides backward compatibility with the SendControlVariateProcessor
        which expects trainer.client_control_variate_delta to be accessible.

        Returns:
            The client control variate delta from the training context,
            or None if not available.
        """
        return self.context.state.get("client_control_variate_delta")
