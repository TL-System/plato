
from plato.config import Config
from  noisy_datasource import NoisyDataSource
import toy_trainer
from callbacks import SetupPseudoLabelCallback, EvalPseudoLabelCallback
from plato.servers import fedavg
from plato.clients import simple

def main():
    if hasattr(Config().data, "noise"):
        datasource = NoisyDataSource
        callbacks = [SetupPseudoLabelCallback, EvalPseudoLabelCallback]
    else:
        datasource = None
        callbacks = None
        
    trainer = toy_trainer.Trainer
    client = simple.Client(datasource=datasource, trainer=trainer, callbacks=callbacks)
    server = fedavg.Server(datasource=datasource, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()


