import asyncio
import logging
import os
import time
import socketio
import asyncio
from plato.config import Config


def timer_run():
    """Starting a timer to connect to the server."""
    timer = Timer()
    logging.info("Starting a Timer.")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(timer.start_timer())


class TimerEvents(socketio.AsyncClientNamespace):
    """ A custom namespace for socketio.AsyncClient. """
    def __init__(self, namespace, plato_timer):
        super().__init__(namespace)
        self.plato_timer = plato_timer

    async def on_connect(self):
        """ Upon a new connection to the server. """
        logging.info("[Timer] Connected to the server.")

    # pylint: disable=protected-access
    async def on_disconnect(self):
        """ Upon a disconnection event. """
        logging.info("[Timer] The server disconnected the connection.")
        os._exit(0)

    async def on_connect_error(self, data):
        """ Upon a failed connection attempt to the server. """
        logging.info("[Timer] A connection attempt to the server failed.")

    async def on_start_timer(self, data):
        """ Upon receiving a start command from the server. """
        logging.info("[Timer] Received a start command from the server.")
        await self.plato_timer.start_the_loop()

class Timer:
    """ Server-side timer for triggering the server. """
    def __init__(self) -> None:
        self.sio = None

    async def start_timer(self) -> None:
        """ Startup function for a timer. """
        if hasattr(Config().algorithm,
                   'cross_silo') and not Config().is_edge_server():
            # Contact one of the edge servers
            raise NotImplementedError
        else:
            await asyncio.sleep(5)
            logging.info("[Timer] Contacting the central server.")

        self.sio = socketio.AsyncClient(reconnection=True)
        self.sio.register_namespace(
            TimerEvents(namespace='/', plato_timer=self))

        if hasattr(Config().server, 'use_https'):
            uri = 'https://{}'.format(Config().server.address)
        else:
            uri = 'http://{}'.format(Config().server.address)

        if hasattr(Config().server, 'port'):
            # If we are not using a production server deployed in the cloud
            if hasattr(Config().algorithm,
                       'cross_silo') and not Config().is_edge_server():
                raise NotImplementedError
            else:
                uri = '{}:{}'.format(uri, Config().server.port)

        logging.info("[Timer] Connecting to the server at %s.", uri)
        await self.sio.connect(uri)
        await self.sio.emit('timer_alive', {'id': -1})

        logging.info("[Timer] Waiting to be started.")
        await self.sio.wait()

    async def start_the_loop(self) -> None:
        tick = -1
        step = -1
        seconds_per_tick = Config().server.async_seconds_per_tick
        ticks_per_step = Config().server.async_ticks_per_step
        start_time = time.time()

        while True:
            # advances the records
            tick += 1
            if tick % ticks_per_step == 0:
                step += 1
                logging.info("[Timer] Step %s starts.", step)

                # does real stuffs
                await self.sio.emit("start_step", {})

            # accounts for the start time drift
            expected_time = tick * seconds_per_tick
            actual_time = time.time() - start_time
            start_time_drift = actual_time - expected_time

            if start_time_drift < seconds_per_tick:
                sleep_time = seconds_per_tick - start_time_drift
                await asyncio.sleep(sleep_time)