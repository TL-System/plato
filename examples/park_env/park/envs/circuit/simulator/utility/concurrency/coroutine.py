import asyncio

__all__ = ['graceful_shutdown', 'graceful_execute']


# Refer: https://stackoverflow.com/questions/30765606/whats-the-correct-way-to-clean-up-after-an-interrupted-event-loop
def graceful_shutdown(loop: asyncio.AbstractEventLoop = None):
    tasks = asyncio.gather(*asyncio.Task.all_tasks(loop=loop), loop=loop, return_exceptions=True)
    tasks.add_done_callback(lambda t: loop.stop())
    tasks.cancel()

    while not tasks.done() and not loop.is_closed():
        loop.run_forever()
    tasks.exception()


def graceful_execute(coroutine, loop=None):
    loop = loop or asyncio.get_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    except KeyboardInterrupt:
        pass
    finally:
        graceful_shutdown(loop)
