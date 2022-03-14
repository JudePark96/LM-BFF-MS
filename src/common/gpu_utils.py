import torch
import torch.distributed

import numpy as np
import os
import random
import logging
import socket


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_gpu_params(params):

    params.device_ids = list(map(int, params.device_ids.split(',')))

    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
    else:
        assert torch.cuda.is_available()

        logger.info("Initializing GPUs")
        if params.n_gpu > 1:
            assert params.local_rank != -1

            params.world_size = int(os.environ["WORLD_SIZE"])
            params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
            params.global_rank = int(os.environ["RANK"])

            # number of nodes / node ID
            params.n_nodes = params.world_size // params.n_gpu_per_node
            params.node_id = params.global_rank // params.n_gpu_per_node
            params.multi_gpu = True

            assert params.n_nodes == int(os.environ["N_NODES"])
            assert params.node_id == int(os.environ["NODE_RANK"])

        # local job (single GPU)
        else:
            assert params.local_rank == -1

            params.n_nodes = 1
            params.node_id = 0
            params.local_rank = 0
            params.global_rank = 0
            params.world_size = 1
            params.n_gpu_per_node = 1
            params.multi_gpu = False

        # sanity checks
        assert params.n_nodes >= 1
        assert 0 <= params.node_id < params.n_nodes, '%d v.s. %d' % (params.node_id, params.n_nodes)
        assert 0 <= params.local_rank <= params.global_rank < params.world_size, '%d v.s. %d v.s. %d' % (params.local_rank, params.global_rank, params.world_size)
        assert params.world_size == params.n_nodes * params.n_gpu_per_node

        # define whether this is the master process / if we are in multi-node distributed mode
        params.is_master = params.node_id == 0 and params.local_rank == 0
        params.multi_node = params.n_nodes > 1

        # summary
        prefix = f"--- Global rank: {params.global_rank} - "
        logger.info(prefix + "Number of nodes: %i" % params.n_nodes)
        logger.info(prefix + "Node ID        : %i" % params.node_id)
        logger.info(prefix + "Local rank     : %i" % params.local_rank)
        logger.info(prefix + "World size     : %i" % params.world_size)
        logger.info(prefix + "GPUs per node  : %i" % params.n_gpu_per_node)
        logger.info(prefix + "Master         : %s" % str(params.is_master))
        logger.info(prefix + "Multi-node     : %s" % str(params.multi_node))
        logger.info(prefix + "Multi-GPU      : %s" % str(params.multi_gpu))
        logger.info(prefix + "Hostname       : %s" % socket.gethostname())

        # set GPU device
        torch.cuda.set_device(params.device_ids[params.local_rank])

        # initialize multi-GPU
        if params.multi_gpu:
            logger.info("Initializing PyTorch distributed")
            torch.distributed.init_process_group(init_method="env://", backend="nccl")
