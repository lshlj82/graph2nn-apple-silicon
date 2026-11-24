#!/usr/bin/env python3
"""Check that ImageNet is installed correctly for graph2nn (and trains on this machine's GPU).

    python tools/check_imagenet.py                 # full ImageNet via pycls/datasets/data/imagenet
    python tools/check_imagenet.py --root ~/datasets.noindex/imagenet100

Steps (each prints OK / WARNING / FAIL):
  1. structure     class folders identical in train/val, image counts
  2. decoding      a random sample of images opens and preprocesses to 3x224x224
  3. labels        a pretrained ResNet-18 (torchvision) classifies 500 val images; ~70% top-1
                   means the val images sit in the right class folders (full ImageNet only;
                   needs internet once to download ~45 MB of weights; skip with --no-pretrained)
  4. loader speed  images/s through graph2nn's own data loader
  5. end to end    ResNet-34 from the graph2nn config learns one real batch on the GPU (CUDA or MPS)
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pycls.config import cfg                             # noqa: E402
from pycls.datasets.imagenet import ImageNet             # noqa: E402

FULL = (1000, 1281167, 50000)
_ok = {'pass': True}


def report(status, msg):
    print('  [{}] {}'.format(status, msg))
    if status == 'FAIL':
        _ok['pass'] = False


def quiet(fn, *a, **k):
    import logging
    logging.disable(logging.INFO)
    try:
        return fn(*a, **k)
    finally:
        logging.disable(logging.NOTSET)


def step_structure(root):
    print('1. structure:', root)
    tr = quiet(ImageNet, root, 'train')
    va = quiet(ImageNet, root, 'val')
    nc, ntr, nva = len(tr._class_ids), len(tr), len(va)
    print('     classes {}, train images {:,}, val images {:,}'.format(nc, ntr, nva))
    if tr._class_ids != va._class_ids:
        report('FAIL', 'train and val class folders differ')
    else:
        report('OK', 'train and val have identical, sorted class folders')
    if nc == 1000:
        if (ntr, nva) == FULL[1:]:
            report('OK', 'counts match ImageNet-1k exactly')
        else:
            report('WARNING', 'expected {:,} train / {:,} val images for ImageNet-1k'.format(*FULL[1:]))
    per_val = np.bincount([r['class'] for r in va._imdb], minlength=nc)
    if per_val.min() == 0:
        report('FAIL', '{} classes have no val images'.format(int((per_val == 0).sum())))
    elif nc == 1000:
        report('OK' if per_val.min() == per_val.max() == 50 else 'WARNING',
               'val images per class: min {} max {} (ImageNet-1k: 50 each)'.format(per_val.min(), per_val.max()))
    return tr, va


def step_decode(tr, va, n=200):
    import cv2
    print('2. decoding {} random images'.format(2 * n))
    rng = random.Random(0)
    bad = []
    for ds in (tr, va):
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            p = ds._imdb[i]['im_path']
            if cv2.imread(p) is None:
                bad.append(p)
    if bad:
        report('FAIL', '{} unreadable images, e.g. {}'.format(len(bad), bad[0]))
    else:
        report('OK', 'all sampled images decode')
    x, y = tr[0]
    xv, _ = va[0]
    ok = x.shape == (3, 224, 224) and xv.shape == (3, 224, 224) and x.dtype == np.float32
    report('OK' if ok else 'FAIL', 'preprocessed sample: {} {}, label {}'.format(x.shape, x.dtype, y))


def step_pretrained(va, n=500):
    print('3. label check with a pretrained ResNet-18 on {} val images'.format(n))
    if len(va._class_ids) != 1000:
        report('OK', 'skipped (only meaningful for the full 1000-class ImageNet)')
        return
    try:
        import cv2
        import torchvision
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=w).eval()
    except Exception as e:
        report('WARNING', 'skipped: could not load pretrained weights ({})'.format(str(e).splitlines()[0][:80]))
        return
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    net = net.to(dev)
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    idx = random.Random(1).sample(range(len(va)), n)
    correct = 0
    with torch.no_grad():
        for b in range(0, n, 50):
            xs, ys = [], []
            for i in idx[b:b + 50]:
                r = va._imdb[i]
                im = cv2.cvtColor(cv2.imread(r['im_path']), cv2.COLOR_BGR2RGB)
                h, w_ = im.shape[:2]; s = 256 / min(h, w_)
                im = cv2.resize(im, (round(w_ * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
                h, w_ = im.shape[:2]; t, l = (h - 224) // 2, (w_ - 224) // 2
                im = (im[t:t + 224, l:l + 224].astype(np.float32) / 255 - mean) / std
                xs.append(im.transpose(2, 0, 1)); ys.append(r['class'])
            pred = net(torch.from_numpy(np.stack(xs)).to(dev)).argmax(1).cpu().numpy()
            correct += int((pred == np.array(ys)).sum())
    acc = 100.0 * correct / n
    # torchvision's class order is the sorted WordNet ids, the same order graph2nn uses
    if acc > 55:
        report('OK', 'top-1 {:.1f}% (expected ~70%): val labels are correct'.format(acc))
    elif acc > 5:
        report('WARNING', 'top-1 {:.1f}%: lower than expected (~70%); some labels may be wrong'.format(acc))
    else:
        report('FAIL', 'top-1 {:.1f}%: val images are NOT in the right class folders'.format(acc))


def step_loader(root, workers, batches=40):
    print('4. loader speed ({} workers, batch 32)'.format(workers))
    ds = quiet(ImageNet, root, 'train')
    ld = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=workers,
                                     persistent_workers=workers > 0, drop_last=True)
    it = iter(ld)
    for _ in range(5):
        next(it)                                          # warm up workers
    t = time.time()
    for _ in range(batches):
        next(it)
    ips = 32 * batches / (time.time() - t)
    report('OK', '{:.0f} images/s  ->  ~{:.0f} min per full-ImageNet epoch from data loading alone'.format(
        ips, 1281167 / ips / 60))
    return ld


def step_end_to_end(root, steps):
    print('5. end to end: ResNet-34 (graph2nn config) learns one real batch')
    from pycls.models import model_builder
    cfg.merge_from_file(os.path.join(ROOT, 'configs/baselines/imagenet/resnet34_imagenet/best/R-34_bs32_1gpu_baseline.yaml'))
    cfg.DATA_LOADER.IMAGENET_PATH = root
    cfg.NUM_GPUS = 1
    model = quiet(model_builder.build_model)
    dev = next(model.parameters()).device
    ds = quiet(ImageNet, root, 'train')
    idx = random.Random(2).sample(range(len(ds)), 32)
    x = torch.from_numpy(np.stack([ds[i][0] for i in idx])).to(dev)
    y = torch.tensor([ds[i][1] for i in idx]).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    model.train()
    losses = []
    t = time.time()
    for s in range(steps):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    dt = (time.time() - t) / steps
    print('     device {}, loss {:.2f} -> {:.2f}, {:.2f} s/step (batch 32)'.format(dev, losses[0], losses[-1], dt))
    report('OK' if losses[-1] < 0.5 * losses[0] else 'FAIL',
           'loss falls on a fixed batch: images, labels, model and GPU are wired correctly')
    print('     note: ~{:.0f} min per full-ImageNet epoch of compute at this speed'.format(1281167 / 32 * dt / 60))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.path.join(ROOT, 'pycls/datasets/data/imagenet'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--steps', type=int, default=30, help='training steps on the fixed batch')
    a = ap.parse_args()
    root = os.path.realpath(os.path.expanduser(a.root))
    if not os.path.isdir(os.path.join(root, 'train')):
        print('FAIL: no train/ folder under {}'.format(root)); sys.exit(1)
    tr, va = step_structure(root)
    step_decode(tr, va)
    if not a.no_pretrained:
        step_pretrained(va)
    step_loader(root, a.workers)
    step_end_to_end(root, a.steps)
    print('\nRESULT: {}'.format('all checks passed' if _ok['pass'] else 'some checks FAILED (see above)'))


if __name__ == '__main__':
    main()
