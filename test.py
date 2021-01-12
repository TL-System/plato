import asyncio
import functools
import sys


def event_handler(loop, num, stop=False):
    print('Event handler called', num)
    if stop:
        print('stopping the loop')
        loop.stop()


async def my_task(seconds):
    """
    A task to do for a number of seconds
    """
    print('This task is taking {} seconds to complete'.format(seconds))
    await asyncio.sleep(seconds)
    return 'task finished'


if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    loop.run_until_complete(my_task(1))

    print('call run forever')
    sys.exit()
    """
    loop.run_forever()

    try:
        loop.call_soon(functools.partial(event_handler, loop, num=1))
        print('starting event loop')

        loop.call_soon(functools.partial(event_handler, loop, num=2,
                                         stop=True))
        print('call run forever')

        loop.run_forever()
    finally:
        print('closing event loop')
        loop.close()
    """