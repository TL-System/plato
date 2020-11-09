import asyncio
import logging
import argparse

import config
from clients import SimpleClient

# Setting up the parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.conf',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s',
    level='INFO', datefmt='%H:%M:%S')


def main():
    """Run a federated learning client."""
    fl_config = config.Config(args.config)

    client = SimpleClient(fl_config)
    client.configure()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start_client())
    loop.run_forever()


if __name__ == "__main__":
    main()
