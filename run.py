import argparse
import client
import config
import logging
import os
import server


# Set up the parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.conf',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set the logging level
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """ Run a simulation for one machine learning training workload using federated learning. """

    # Read runtime parameters from a configuration file
    fl_config = config.Config(args.config)

    # Initialize the federated learning server

if __name__ == "__main__":
    main()
