import asyncio
import logging
import argparse
import websockets

import config
from clients import SimpleClient

# Setting up the parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', type=str,
                    help='Unique client ID.')
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

    client = SimpleClient(fl_config, args.id)
    client.configure()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(client.start_client())
    except websockets.ConnectionClosed:
        logging.info("Client #%s: connection to the server is closed.",
            client.client_id)


if __name__ == "__main__":
    main()
