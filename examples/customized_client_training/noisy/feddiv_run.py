
from plato.config import Config
from  noisy_datasource import NoisyDataSource
import toy_trainer
from callbacks import SetupPseudoLabelCallback, EvalPseudoLabelCallback
from feddiv import fd_server, fd_client, fd_trainer
from plato.clients import simple

def main():
    if hasattr(Config().data, "noise"):
        datasource = NoisyDataSource
    else:
        datasource = None

    trainer = fd_trainer.Trainer
    client = fd_client.Client(datasource=datasource, trainer=trainer)
    server = fd_server.Server(datasource=datasource, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()


