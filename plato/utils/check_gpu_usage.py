import pynvml as pynvml
import psutil

def check_gpu_usage(process_exceptions=['Xorg'], user_exceptions=['bla123'], min_memory=5, base_on_memory=True, base_on_process=True):
    # Process exceptions -> we don't care about such procs
    # User exceptions -> we care ONLY about procs of this user
    pynvml.nvmlInit()
    # print ("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
    deviceCount = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    for i in range(deviceCount):

        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem.free/(1024**3)
        if base_on_memory and free_memory < min_memory:
            continue

        free = True 
        if base_on_process:
            procs = [*pynvml.nvmlDeviceGetComputeRunningProcesses(handle), *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]
            for p in procs:
                try:
                    process = psutil.Process(p.pid)
                except psutil.NoSuchProcess:
                    continue

                if process.name not in process_exceptions and process.username() in user_exceptions:
                    free = False
                    break
        if free:
            free_gpus.append(str(i))

    print(f"[[GPU INFO]] [{', '.join(free_gpus)}] are free and made visible to this session.")
    pynvml.nvmlShutdown()
    return ','.join(free_gpus)
