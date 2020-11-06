"""
Running the Plato federated learning emulator.
"""

import asyncio
import json
import logging
import argparse
import os
import websockets

import config
import servers

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

clients = {}

async def register(client_id, websocket):
    if not client_id in clients:
        clients[client_id] = websocket

    logging.info("clients: %s", clients)


async def unregister(websocket):
    for key, value in dict(clients).items():
        if value == websocket:
            del clients[key]
    logging.info("clients: %s", clients)


async def fl_server(websocket, path):
    try:
        async for message in websocket:
            data = json.loads(message)
            client_id = data["id"]
            await register(client_id, websocket)
            logging.info("client received with ID: %s", client_id)

            response = {'id': client_id}
            await websocket.send(json.dumps(response))
    finally:
        await unregister(websocket)


def main():
    """Run a federated learning server."""
    # Read runtime parameters from a configuration file
    fl_config = config.Config(args.config)

    # Initialize the federated learning server
    server = {
        "fedavg": servers.fedavg.FedAvgServer
    }[fl_config.training.server](fl_config)
    server.configure()

    logging.info("Starting the federated learning server...")
    start_server = websockets.serve(fl_server, "localhost", 8000)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

if __name__ == "__main__":
    main()
