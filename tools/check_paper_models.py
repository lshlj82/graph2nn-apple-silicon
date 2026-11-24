#!/usr/bin/env python3
"""Check that this port builds the same models as the paper, without training.

For every model family of the paper and every graph whose raw results ship with the repository
(analysis/final_results/), build the model from the paper's config, run the original FLOPs matching
(TRAIN.AUTO_MATCH), and compare its parameter count and FLOPs with the values the original authors
recorded for that run. Runs on the CPU and needs no dataset.

    python tools/check_paper_models.py                      # all families, baseline + best graph
    python tools/check_paper_models.py --all                # the 52 (EfficientNet: 48) graphs of the paper's
                                                            # cross-task comparison, per family (1-2 h on a CPU)
    python tools/check_paper_models.py --task resnet34_imagenet --all

The paper's values are rounded to 0.001 M parameters and 0.001 GFLOPs, so a model passes when both
counts agree within that rounding.
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import train_net                                        # noqa: E402  (match_complexity, original code)
from pycls.config import cfg                            # noqa: E402
from pycls.models import model_builder                  # noqa: E402
import pycls.utils.metrics as mu                        # noqa: E402

# task: (config folder, 8-GPU config prefix as used in the paper, raw-results folder prefix)
TASKS = {
    'mlp_cifar10': ('cifar10/mlp_cifar10', 'mlp_bs128_1gpu_layer3', 'mlp_cifar10'),
    'cnn_imagenet': ('imagenet/cnn_imagenet', 'cnn6_bs256_8gpu_64d', 'cnn6_imagenet'),
    'resnet34_imagenet': ('imagenet/resnet34_imagenet', 'R-34_bs256_8gpu', 'resnet34_imagenet'),
    'resnet34sep_imagenet': ('imagenet/resnet34sep_imagenet', 'R-34_bs256_8gpu', 'resnet34_sep_imagenet'),
    'resnet50_imagenet': ('imagenet/resnet50_imagenet', 'R-50_bs256_8gpu', 'resnet50_imagenet'),
    'efficient_imagenet': ('imagenet/efficient_imagenet', 'EN-B0_bs512_8gpu_nms', 'efficientnet_imagenet'),
}
_PAT = re.compile(r'_sparsity([\d.]+)_p([\d.]+)_agg\w+?_graphseed(\d+)_')
DEFAULTS = cfg.clone()


def paper_values(task):
    """{(sparsity, p, graphseed): (params M, GFLOPs)} from the original raw results (seed 1)."""
    pre = TASKS[task][2]
    if task == 'mlp_cifar10':
        d = os.path.join(ROOT, 'analysis/final_results/mlp_cifar10')
        names = np.load(glob.glob(os.path.join(d, 'names_*epoch200.npy'))[0], allow_pickle=True)
        res = np.load(glob.glob(os.path.join(d, 'results_*epoch200.npy'))[0], allow_pickle=True)
    else:
        d = os.path.join(ROOT, 'analysis/final_results/{}_seed1'.format(pre))
        names = np.load(glob.glob(os.path.join(d, 'names_*epoch100.npy'))[0], allow_pickle=True)
        res = np.load(glob.glob(os.path.join(d, 'results_*epoch100.npy'))[0], allow_pickle=True)
    out = {}
    for n, r in zip(names, res):
        r = np.asarray(r, dtype=float).reshape(-1, 8)[0]
        key = ('linear',) if n[0] == 'linear' else (float(n[4]), float(n[5]), int(float(n[7])))
        out[key] = (r[0], r[1])
    return out


def config_files(task, everything):
    folder, prefix, _ = TASKS[task]
    base = os.path.join(ROOT, 'configs/baselines', folder)
    best = sorted(f for f in glob.glob(os.path.join(base, 'best', '*.yaml')) if prefix in os.path.basename(f))
    if not everything:
        return best
    g = np.load(os.path.join(ROOT, 'analysis', 'graphs_n16_48.npy' if task == 'efficient_imagenet' else 'graphs_n64_52.npy'))
    keep = []
    for f in sorted(glob.glob(os.path.join(base, 'all', '*.yaml'))):
        m = _PAT.search(os.path.basename(f))
        if prefix in os.path.basename(f) and m and any(
                abs(r[1] - float(m.group(1))) < 2e-6 and abs(r[2] - float(m.group(2))) < 2e-6 and int(r[3]) == int(m.group(3))
                for r in g):
            keep.append(f)
    return best + keep


def key_of():
    if cfg.RESNET.TRANS_FUN == 'linear_transform':
        return ('linear',)
    return (round(cfg.RGRAPH.SPARSITY, 6), round(cfg.RGRAPH.P, 6), int(cfg.RGRAPH.SEED_GRAPH))


def lookup(paper, key):
    if key in paper:
        return paper[key]
    for k, v in paper.items():
        if len(k) == 3 and len(key) == 3 and abs(k[0] - key[0]) < 2e-6 and abs(k[1] - key[1]) < 2e-6 and k[2] == key[2]:
            return v
    return None


def build(cfg_file):
    cfg.merge_from_other_cfg(DEFAULTS)
    cfg.merge_from_file(cfg_file)
    cfg.NUM_GPUS = 1
    cfg.RGRAPH.SEED_TRAIN = cfg.RGRAPH.SEED_TRAIN_START
    train_net._CFG_FILE = os.path.basename(cfg_file)
    cfg.OUT_DIR = '.'
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        train_net.match_complexity()
        model = model_builder.build_model()
    return mu.params_count(model) / 1e6, mu.flops_count(model) / 1e9, list(cfg.RGRAPH.DIM_LIST)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', nargs='+', default=list(TASKS), choices=list(TASKS))
    ap.add_argument('--all', action='store_true', help='every graph with paper results, not only baseline + best')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    total_fail = 0
    for task in a.task:
        paper = paper_values(task)
        files = config_files(task, a.all)
        seen, ok, fail, skip = set(), 0, 0, 0
        t0 = time.time()
        print('\n=== {} ({} configs) ==='.format(task, len(files)))
        for f in files:
            cfg.merge_from_other_cfg(DEFAULTS)
            cfg.merge_from_file(f)
            key = key_of()
            if key in seen:
                continue
            seen.add(key)
            ref = lookup(paper, key)
            if ref is None:
                skip += 1
                continue
            params, flops, dims = build(f)
            good = abs(params - ref[0]) <= 0.0015 and abs(flops - ref[1]) <= 0.0015
            ok += good
            fail += not good
            label = 'baseline' if (key == ('linear',) or key[0] >= 1) else 'graph s={:.4f} p={:.4f} seed={}'.format(*key)
            if not a.all or not good:
                print('  [{}] {:<42s} params {:8.3f} M (paper {:8.3f})  FLOPs {:7.3f} G (paper {:7.3f})  dims {}'.format(
                    'OK' if good else 'FAIL', label, params, ref[0], flops, ref[1], dims))
        print('  {}: {} OK, {} FAIL{}  ({:.0f} s)'.format(task, ok, fail, ', {} without paper results'.format(skip) if skip else '',
                                                          time.time() - t0))
        total_fail += fail
    print('\nRESULT: {}'.format('all models match the paper' if total_fail == 0 else '{} models differ'.format(total_fail)))
    sys.exit(1 if total_fail else 0)


if __name__ == '__main__':
    main()
