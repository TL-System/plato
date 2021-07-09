import logging
import os

# need to run 'ulimit -n 64000' on the server nodes
os.system('ulimit -n 64000')
os.environ['config_file'] = 'examples/dist_mistnet/mistnet_lenet5_server.yml'
from plato.servers import mistnet

class CustomServer(mistnet.Server):
    """ A custom federated learning server. """
    def __init__(self, model=None, trainer=None):
        super().__init__()
        logging.info("A custom server has been initialized.")

def main():
    """ A Plato federated learning training session using a custom model. """
    server = CustomServer()
    server.run()

if __name__ == "__main__":
    main()
