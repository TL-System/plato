
from plato.config import Config
from  noisy_datasource import NoisyDataSource
import noisy_trainer 
from plato.servers import fedavg
from plato.clients import simple

def main():
    if hasattr(Config().data, "noise"):
        datasource = NoisyDataSource
    else:
        datasource = None
        
    trainer = noisy_trainer.Trainer
    client = simple.Client(datasource=datasource, trainer=trainer)
    server = fedavg.Server(datasource=datasource, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()


