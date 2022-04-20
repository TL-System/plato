import pynvml as pynvml
import psutil

def available_gpu():
    # Process exceptions -> we don't care about such procs
    # User exceptions -> we care ONLY about procs of this user
    pynvml.nvmlInit()
    # print ("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
    deviceCount = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    min_memory_used = 0
    device_id = -1 

    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem.free / (1024**3)

        if free_memory < min_memory_used:
            continue
        else:
            min_memory_used = free_memory
            device_id = i

    print(f"[GPU INFO] [{device_id}] has the most available memory and made visible to this session.")
    pynvml.nvmlShutdown()
    return str(device_id)
