#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Device selection helpers (CUDA / Apple-Silicon MPS / CPU)."""

import os

import torch

from pycls.config import cfg

_DEVICE = None


def _mps_available():
    return getattr(torch.backends, 'mps', None) is not None and \
        torch.backends.mps.is_available()


def get_device():
    """Returns the torch.device selected by cfg.DEVICE.

    cfg.DEVICE may be 'auto' (CUDA > MPS > CPU), 'cuda', 'mps' or 'cpu'.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    name = cfg.DEVICE.lower()
    if name == 'auto':
        if torch.cuda.is_available():
            name = 'cuda'
        elif _mps_available():
            name = 'mps'
        else:
            name = 'cpu'
    if name == 'cuda':
        assert torch.cuda.is_available(), 'cfg.DEVICE=cuda but CUDA is not available'
        _DEVICE = torch.device('cuda', torch.cuda.current_device())
    elif name == 'mps':
        assert _mps_available(), 'cfg.DEVICE=mps but MPS is not available'
        _DEVICE = torch.device('mps')
    elif name == 'cpu':
        _DEVICE = torch.device('cpu')
    else:
        raise ValueError('Unknown cfg.DEVICE: {}'.format(cfg.DEVICE))
    return _DEVICE


def reset_device():
    """Forget the cached device (e.g. after torch.cuda.set_device in a worker)."""
    global _DEVICE
    _DEVICE = None


def is_cuda():
    return get_device().type == 'cuda'


def is_mps():
    return get_device().type == 'mps'


def num_available_devices():
    """Number of accelerators usable for data parallel training."""
    dev = get_device()
    if dev.type == 'cuda':
        return torch.cuda.device_count()
    if dev.type == 'cpu':
        return os.cpu_count() or 1  # multi-process CPU training (gloo), for testing
    return 1  # MPS exposes a single device


def synchronize():
    dev = get_device()
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    elif dev.type == 'mps':
        torch.mps.synchronize()


def max_memory_allocated():
    """Peak (CUDA) / current driver (MPS) memory in bytes; 0 on CPU."""
    dev = get_device()
    if dev.type == 'cuda':
        return torch.cuda.max_memory_allocated()
    if dev.type == 'mps':
        return torch.mps.driver_allocated_memory()
    return 0
