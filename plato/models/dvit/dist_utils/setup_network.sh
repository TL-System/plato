rdma_capable='ibv_devinfo 2>&1 |grep "mlx5_0" |wc -l'
if [ $rdma_capable -gt 0 ]; then
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1
    export NCCL_IB_HCA=mlx5_0:1
    export NCCL_IB_GID_INDEX=3
    export NCCL_SCOKET_IFNAME=eth0
    export HOROVOD_MPI_THREADS_DISABLE=1
else
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=eth0
fi
