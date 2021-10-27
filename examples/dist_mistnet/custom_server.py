import os

os.environ['config_file'] = './mistnet_lenet5_server.yml'
from plato.servers import mistnet

def main():
    """ A Plato federated learning server using the MistNet algorithm. """
    server = mistnet.Server()
    server.run()

if __name__ == "__main__":
    main()
