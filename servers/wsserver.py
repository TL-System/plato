import asyncio
import json
import logging
import websockets

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
    logging.info("Path: %s", path)
    
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


start_server = websockets.serve(fl_server, "localhost", 8000)

loop = asyncio.get_event_loop()
loop.run_until_complete(start_server)
loop.run_forever()