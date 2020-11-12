"""
Starting point for a Plato federated learning client.
"""

import asyncio
import logging
import argparse
import websockets

import config
from clients import SimpleClient


def main():
    """Starting a client to connect to the server via WebSockets."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', type=str,
                        help='Unique client ID.')
    parser.add_argument('-c', '--config', type=str, default='./config.conf',
                        help='Federated learning configuration file.')
    parser.add_argument('-l', '--log', type=str, default='INFO',
                        help='Log messages level.')

    args = parser.parse_args()

    try:
        log_level = {
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warn': logging.WARN,
            'info': logging.INFO,
            'debug': logging.DEBUG
        }[args.log]
    except KeyError:
        log_level = logging.INFO

    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s',
        level=log_level, datefmt='%H:%M:%S')
    logging.info("Log level: %s", args.log)

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
