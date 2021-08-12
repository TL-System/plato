import logging
import os

# need to run 'ulimit -n 64000' on the server nodes
# os.system('ulimit -n 64000')
os.environ['config_file'] = 'examples/dist_mistnet/mistnet_lenet5_server.yml'
from plato.servers import mistnet
from plato.utils import transmitter

class CustomServer(mistnet.Server):
    """ A custom federated learning server. """
    def __init__(self, model=None, trainer=None, transmitter=None):
        super().__init__(transmitter=transmitter)
        logging.info("A custom server has been initialized.")

def main():
    """ A Plato federated learning training session using a custom model. """
    trans = transmitter.S3Transmitter("https://obs.cn-south-1.myhuaweicloud.com", "EKPTZ0OPJC4SRAHPTZCA", "LiBjVWjbiVs37eiY9IdZ0OVnlBY4T3hBVgywaE9D", "plato")
    server = CustomServer(transmitter = trans)
    server.run()

if __name__ == "__main__":
    main()
