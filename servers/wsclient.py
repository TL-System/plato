import asyncio
import uuid
import json
import websockets

async def hello():
    uri = "ws://localhost:8000"
    async with websockets.connect(uri) as websocket:
        client_id = uuid.uuid1().int
        print(f"{client_id} generated.")

        await websocket.send(json.dumps({'id': client_id}))
        print(f"> {client_id}")

        response = await websocket.recv()
        data = json.loads(response)
        print(f"< {data['id']}")

asyncio.get_event_loop().run_until_complete(hello())
