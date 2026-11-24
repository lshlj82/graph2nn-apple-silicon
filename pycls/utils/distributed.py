#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed helpers."""

import os

import torch

from pycls.config import cfg
import pycls.utils.device as dev


def is_master_proc():
    """Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints.
    In the multi GPU setting, we assign the master role to the rank 0 process.
    When training using a single GPU, there is only one training processes
    which is considered the master processes.
    """
    return cfg.NUM_GPUS == 1 or torch.distributed.get_rank() == 0


def init_process_group(proc_rank, world_size):
    """Initializes the default process group."""
    # Multi-process training runs on CUDA (one GPU per process); cfg.DEVICE cpu runs it on the CPU with
    # the gloo backend (for testing). MPS exposes one device and is not supported.
    if cfg.DEVICE.lower() == 'cpu':
        backend = 'gloo'
    else:
        assert torch.cuda.is_available(), \
            'NUM_GPUS > 1 requires CUDA; on Apple Silicon (MPS) use NUM_GPUS 1'
        torch.cuda.set_device(proc_rank)
        backend = cfg.DIST_BACKEND
    dev.reset_device()
    os.environ.setdefault('MASTER_ADDR', cfg.HOST)
    os.environ.setdefault('MASTER_PORT', str(cfg.PORT))
    # Initialize the process group
    # print('----------rank{},world{}---------'.format(proc_rank, world_size))
    # torch.distributed.init_process_group(
    #     backend=cfg.DIST_BACKEND,
    #     init_method="tcp://{}:{}".format(cfg.HOST, cfg.PORT),
    #     world_size=world_size,
    #     rank=proc_rank
    # )
    torch.distributed.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=proc_rank
    )


def destroy_process_group():
    """Destroys the default process group."""
    torch.distributed.destroy_process_group()


def scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of
    the process group (equivalent to cfg.NUM_GPUS).
    """
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.NUM_GPUS)
    return tensors
