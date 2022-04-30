'''
    A handy utility that looks for the best GPU with the most free available
    CUDA memory.
'''

import logging
import pynvml


def available_gpu() -> int:
    ''' Returns the best GPU with the most free available CUDA memory. '''
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    max_memory_unused = 0
    device_id = -1

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem.free / (1024**3)

        if free_memory > max_memory_unused:
            max_memory_unused = free_memory
            device_id = i
        else:
            continue

    logging.info("GPU #%d is used as it has the most available memory.",
                 device_id)
    pynvml.nvmlShutdown()

    return device_id
