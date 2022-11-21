#!/usr/bin/env bash

if {{ -z "${ARNOLD_ID}" ]]; then
    echo "Not on Arnold"
    export METIS_TASK_INDEX=0
    export METIS_WORKER_0_HOST=localhost
    export METIS_WORKER_0_PORT=9000
    export ARNOLD_WORKER_NUM=1
    # export ARNOLD_OUTPUT=exp
else
    echo "On Arnold"
    source ./dist_utils/setup_network.sh
fi

# export ARNOLD_OUTPUT=/mnt/cephfs_new_wj/uslabcv/zhoudaquan/ss_design/ea_output
export DATA_IMG=/mnt/cephfs_new_wj/uslabcv/ultron/datasets/ILSVRC/Data/CLS-LOC/

echo METIS_CONFIG: $METIS_WORKER_0_HOST:$METIS_WORKER_0_PORT, $METIS_TASK_INDEX
echo NUM_WORKER: $ARNOLD_WORKER_NUM, WORKER_GPU: $ARNOLD_WORKER_GPU
# echo OUTPUT_DIR: ${ARNOLD_OUTPUT}
