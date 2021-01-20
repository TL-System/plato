import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

def train(rank, num_epochs, world_size):
    init_process(rank, world_size)
    print(
        f"Rank {rank + 1}/{world_size} process initialized.\n"
    )
    # rest of the training script goes here!

WORLD_SIZE = torch.cuda.device_count()

if __name__=="__main__":
    mp.spawn(
        train, args=(10, WORLD_SIZE),
        nprocs=WORLD_SIZE, join=True
    )
