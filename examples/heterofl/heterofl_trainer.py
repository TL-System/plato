"""
HeteroFL algorithm trainer.
"""

from plato.trainers.basic import Trainer

class ServerTrainer(Trainer):
    """A federated learning trainer of Hermes, used by the server."""
    def test(self, testset, sampler=None, **kwargs) -> float:
        """Because the global model will need to compute the statistics of the model."""
        self.train(testset,sampler,**kwargs)
        return super().test(testset, sampler, **kwargs)
    