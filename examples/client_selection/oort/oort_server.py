"""
A federated learning server using Oort.

Reference:

F. Lai, X. Zhu, H. V. Madhyastha and M. Chowdhury, "Oort: Efficient Federated Learning via
Guided Participant Selection," in USENIX Symposium on Operating Systems Design and Implementation
(OSDI 2021), July 2021.
"""

from oort_selection_strategy import OortSelectionStrategy

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the Oort client selection strategy."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        server_config = Config().server

        strategy = OortSelectionStrategy(
            exploration_factor=server_config.exploration_factor,
            desired_duration=server_config.desired_duration,
            step_window=server_config.step_window,
            penalty=server_config.penalty,
            cut_off=getattr(server_config, "cut_off", 0.95),
            blacklist_num=getattr(server_config, "blacklist_num", 10),
        )

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            client_selection_strategy=strategy,
        )
