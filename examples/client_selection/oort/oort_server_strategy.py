"""
A federated learning server using Oort with strategy pattern.

This is the updated version using the strategy-based API instead of inheritance.

Reference:

F. Lai, X. Zhu, H. V. Madhyastha and M. Chowdhury, "Oort: Efficient Federated Learning via
Guided Participant Selection," in USENIX Symposium on Operating Systems Design and Implementation
(OSDI 2021), July 2021.
"""

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies import OortSelectionStrategy


class Server(fedavg.Server):
    """A federated learning server using the Oort client selection strategy."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        # Load Oort parameters from config
        exploration_factor = Config().server.exploration_factor
        desired_duration = Config().server.desired_duration
        step_window = Config().server.step_window
        penalty = Config().server.penalty

        cut_off = (
            Config().server.cut_off if hasattr(Config().server, "cut_off") else 0.95
        )

        blacklist_num = (
            Config().server.blacklist_num
            if hasattr(Config().server, "blacklist_num")
            else 10
        )

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            client_selection_strategy=OortSelectionStrategy(
                exploration_factor=exploration_factor,
                desired_duration=desired_duration,
                step_window=step_window,
                penalty=penalty,
                cut_off=cut_off,
                blacklist_num=blacklist_num,
            ),
        )
