
from plato.config import Config
from  noisy_datasource import NoisyDataSource
import toy_trainer
from fedcorr import fc_server, fc_client, fc_trainer
from plato.clients import simple

def main():
    if hasattr(Config().data, "noise"):
        datasource = NoisyDataSource
    else:
        datasource = None
    
    trainer = fc_trainer.Trainer
    client = fc_client.Client(datasource=datasource, trainer=trainer)
    server = fc_server.Server(datasource=datasource, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()


