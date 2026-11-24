#!/usr/bin/env python3
"""Quick sanity check that graph2nn models train on Apple-Silicon MPS.

Runs a few SGD steps of a relational-graph model on CPU and on MPS from the
same initial weights and compares the losses.

    python tools/check_mps.py
    python tools/check_mps.py --cfg configs/baselines/imagenet/resnet34_imagenet/best/R-34_bs32_1gpu_best.yaml
"""

import argparse
import copy
import time

import torch

from pycls.config import cfg
from pycls.models import model_builder
import pycls.utils.device as dev

p = argparse.ArgumentParser()
p.add_argument('--cfg', default='configs/baselines/cifar10/mlp_cifar10/best/'
                                'mlp_bs128_1gpu_layer3_best.yaml')
p.add_argument('--steps', type=int, default=5)
p.add_argument('--bs', type=int, default=32)
a = p.parse_args()

assert torch.backends.mps.is_available(), \
    'MPS not available: use a native arm64 Python and torch>=2.0 on macOS 12.3+'
cfg.merge_from_file(a.cfg)
cfg.NUM_GPUS = 1
cfg.DEVICE = 'cpu'
torch.manual_seed(0)
model_cpu = model_builder.build_model()
model_mps = copy.deepcopy(model_cpu).to('mps')
size = 32 if cfg.TRAIN.DATASET == 'cifar10' else 224
x = torch.randn(a.bs, 3, size, size)
y = torch.randint(0, cfg.MODEL.NUM_CLASSES, (a.bs,))

for name, model, d in [('cpu', model_cpu, 'cpu'), ('mps', model_mps, 'mps')]:
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    xs, ys = x.to(d), y.to(d)
    t = time.time()
    losses = []
    for _ in range(a.steps):
        loss = torch.nn.functional.cross_entropy(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print('{:4s} losses: {}  ({:.2f}s)'.format(
        name, ' '.join('{:.4f}'.format(l) for l in losses), time.time() - t))
masks = [b.device.type for n, b in model_mps.named_buffers() if n.endswith('mask')]
print('graph-mask buffers on MPS: {}/{}'.format(masks.count('mps'), len(masks)))
