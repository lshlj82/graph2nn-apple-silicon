#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model construction functions."""

import torch

from pycls.config import cfg
from pycls.models.resnet import ResNet
from pycls.models.mlp import MLPNet
from pycls.models.cnn import CNN
from pycls.models.mobilenet import MobileNetV1
from pycls.models.efficientnet import EfficientNet
from pycls.models.vgg import VGG

import pycls.utils.device as dev
import pycls.utils.logging as lu
import pycls.utils.metrics as mu

logger = lu.get_logger(__name__)

# Supported model types
_MODEL_TYPES = {
    'resnet': ResNet,
    'mlpnet': MLPNet,
    'cnn': CNN,
    'mobilenet': MobileNetV1,
    'efficientnet': EfficientNet,
    'vgg': VGG,
}


def build_model():
    """Builds the model."""
    assert cfg.MODEL.TYPE in _MODEL_TYPES.keys(), \
        'Model type \'{}\' not supported'.format(cfg.MODEL.TYPE)
    assert cfg.NUM_GPUS <= dev.num_available_devices(), \
        'Cannot use more GPU devices than available'
    # Construct the model
    model = _MODEL_TYPES[cfg.MODEL.TYPE]()
    # Determine the device used by the current process (CUDA / MPS / CPU)
    cur_device = dev.get_device()
    # Transfer the model (and its graph-mask buffers) to the device
    model = model.to(cur_device)
    # Use multi-process data parallel model in the multi-gpu setting (CUDA only)
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        cuda = cur_device.type == 'cuda'
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device.index] if cuda else None,
            output_device=cur_device.index if cuda else None
        )
    return model


## auto match flop
def build_model_stats(mode='flops'):
    """Builds the model."""
    assert cfg.MODEL.TYPE in _MODEL_TYPES.keys(), \
        'Model type \'{}\' not supported'.format(cfg.MODEL.TYPE)
    # Construct the model (kept on CPU; only used for counting)
    model = _MODEL_TYPES[cfg.MODEL.TYPE]()
    if mode == 'flops':
        flops = mu.flops_count(model)
        return flops
    else:
        params = mu.params_count(model)
        return params
