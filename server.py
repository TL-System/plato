"""
Starting point for a Plato federated learning server.
"""

import asyncio
import logging
import argparse
import websockets

import config
import servers

def main():
    """Starting a WebSockets server."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config.conf',
                        help='Federated learning configuration file.')
    parser.add_argument('-l', '--log', type=str, default='info',
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

    fl_config = config.Config(args.config)

    server = {
        "fedavg": servers.fedavg.FedAvgServer
    }[fl_config.training.server](fl_config)

    server.start_clients()

    start_server = websockets.serve(server.serve,
                     fl_config.server.address, fl_config.server.port,
                     max_size=2 ** 30)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

if __name__ == "__main__":
    main()
