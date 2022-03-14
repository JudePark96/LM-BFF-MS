#!/bin/bash
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=4
export WORLD_SIZE=4

export MASTER_PORT=6692
export MASTER_ADDR="localhost"

DEVICE_IDS=0,1,2,3

CONFIG_PATH=../../rsc/experiment_configs/zeroshot/t5/snli-zeroshot.json

python -m torch.distributed.launch \
          --nproc_per_node ${N_GPU_NODE} \
          --nnodes ${N_NODES} \
          --node_rank ${NODE_RANK} \
          --master_addr ${MASTER_ADDR} \
          --master_port ${MASTER_PORT} \
          t5_main.py --n_gpu ${WORLD_SIZE} \
                   --device_ids ${DEVICE_IDS} \
                   --config_path ${CONFIG_PATH}