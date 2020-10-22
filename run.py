"""
Running the Plato federated learning emulator.
"""

import argparse
import logging
import config

from servers import fedavg

# Set up the parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.conf',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set the logging level
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s',
    level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """ Run an emulation for one machine learning training workload using federated learning. """

    # Read runtime parameters from a configuration file
    fl_config = config.Config(args.config)

    # Initialize the federated learning server
    fl_server = {
        "fedavg": fedavg.Server(fl_config)
    }[fl_config.general.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

if __name__ == "__main__":
    main()
