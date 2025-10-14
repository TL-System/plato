"""
A federated learning server using FedAsync with strategy pattern.

This is the updated version using the strategy-based API instead of inheritance.

Reference:

Xie, C., Koyejo, S., Gupta, I. "Asynchronous federated optimization,"
in Proc. 12th Annual Workshop on Optimization for Machine Learning (OPT 2020).

https://opt-ml.org/papers/2020/paper_28.pdf
"""

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies import FedAsyncAggregationStrategy


class Server(fedavg.Server):
    """A federated learning server using the FedAsync aggregation strategy."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        # Load FedAsync parameters from config
        mixing_hyperparameter = 1
        adaptive_mixing = False
        staleness_func_type = "constant"
        staleness_func_params = {}

        if hasattr(Config().server, "mixing_hyperparameter"):
            mixing_hyperparameter = Config().server.mixing_hyperparameter

        if hasattr(Config().server, "adaptive_mixing"):
            adaptive_mixing = Config().server.adaptive_mixing

        if hasattr(Config().server, "staleness_weighting_function"):
            staleness_func_param = Config().server.staleness_weighting_function
            staleness_func_type = staleness_func_param.type.lower()

            if staleness_func_type == "polynomial" and hasattr(
                staleness_func_param, "a"
            ):
                staleness_func_params["a"] = staleness_func_param.a
            elif staleness_func_type == "hinge":
                if hasattr(staleness_func_param, "a"):
                    staleness_func_params["a"] = staleness_func_param.a
                if hasattr(staleness_func_param, "b"):
                    staleness_func_params["b"] = staleness_func_param.b

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=FedAsyncAggregationStrategy(
                mixing_hyperparameter=mixing_hyperparameter,
                adaptive_mixing=adaptive_mixing,
                staleness_func_type=staleness_func_type,
                staleness_func_params=staleness_func_params,
            ),
        )
