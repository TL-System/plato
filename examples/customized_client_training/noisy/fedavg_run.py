
from plato.config import Config
from  noisy_datasource import NoisyDataSource
from callbacks import SetupPseudoLabelCallback
from plato.servers import fedavg
from plato.clients import simple

def main():
    if hasattr(Config().data, "noise"):
        datasource = NoisyDataSource
        callbacks = [SetupPseudoLabelCallback]
    else:
        datasource = None
        callbacks = None
        
    client = simple.Client(datasource=datasource, callbacks=callbacks)
    server = fedavg.Server(datasource=datasource)
    server.run(client)

if __name__ == "__main__":
    main()


