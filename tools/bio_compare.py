#!/usr/bin/env python3
"""Biological relational graphs (paper Table 2): your results vs. the paper, and one machine vs. another.

The 5-layer MLP on CIFAR-10 with the complete graph and three cortical connectomes (cat, macaque visual,
macaque whole cortex), 5 seeds each:

    bash launch_parallel.sh mlp_cifar10_bio best 1 <JOBS> 0          # results in checkpoint/.../mlp_cifar10_bio/best
    python tools/bio_compare.py                                        # every folder found under mlp_cifar10_bio/
    python tools/bio_compare.py --dirs best best_cuda                  # e.g. Mac vs. results copied from a CUDA server

For each graph: top-1 test error (mean +- std over seeds, as in the paper), the difference to the complete graph,
and the training time per epoch from the logs (time per step x steps per epoch; testing not included). With two
folders, the per-graph difference between them is tested against seed noise (Welch t-test).
"""

import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'checkpoint/cifar10/mlp_cifar10_bio')
# graph: (label, paper mean, paper std)  -- You et al., ICML 2020, Table 2
PAPER = {'ws_sparsity1.0': ('complete graph', 33.34, 0.36),
         'catraw': ('cat cortex', 33.01, 0.22),
         'mcvisualraw': ('macaque visual cortex', 32.78, 0.21),
         'mcwholeraw': ('macaque whole cortex', 32.77, 0.14)}
ORDER = ['ws_sparsity1.0', 'catraw', 'mcvisualraw', 'mcwholeraw']
STEPS_PER_EPOCH = 391          # 50,000 images, batch 128


def graph_of(name):
    m = re.search(r'_message(\w+?)_sparsity([\d.]+)_', name)
    if not m:
        return None
    return 'ws_sparsity' + m.group(2) if m.group(1) == 'ws' else m.group(1)


def load(folder):
    out = {}
    for d in sorted(glob.glob(os.path.join(folder, '*'))):
        g = graph_of(os.path.basename(d))
        f = os.path.join(d, 'results_epoch200.txt')
        if g is None or not os.path.isfile(f):
            continue
        r = np.loadtxt(f, delimiter=',', ndmin=2)
        err = np.array(list({int(row[7]): row[3] for row in r}.values()))
        times = []
        for log in glob.glob(os.path.join(d, 'run_seed*.log')) + glob.glob(os.path.join(d, 'run.log')):
            t = []
            for line in open(log, errors='ignore'):
                if '"_type": "train_epoch"' in line:
                    try:
                        t.append(json.loads(line.split('json_stats: ', 1)[1])['time_avg'])
                    except (ValueError, KeyError, IndexError):
                        pass
            if t:
                times.append(np.median(t) * STEPS_PER_EPOCH)
        out[g] = (err, np.array(times))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dirs', nargs='+', help='folders under checkpoint/cifar10/mlp_cifar10_bio (default: all)')
    a = ap.parse_args()
    dirs = a.dirs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(BASE, '*')) if os.path.isdir(p))
    if not dirs:
        print('no results under {}'.format(os.path.relpath(BASE, ROOT))); return
    res = {d: load(os.path.join(BASE, d)) for d in dirs}

    print('Top-1 test error (%), mean +- std over seeds (n); paper: You et al. 2020, Table 2\n')
    head = '{:<24s} {:>15s}'.format('graph', 'paper')
    for d in dirs:
        head += ' {:>20s} {:>8s}'.format(d, 'vs comp')
    print(head)
    for g in ORDER:
        label, pm, ps = PAPER[g]
        line = '{:<24s} {:>15s}'.format(label, '{:.2f} +- {:.2f}'.format(pm, ps))
        for d in dirs:
            if g in res[d] and len(res[d][g][0]):
                e = res[d][g][0]
                ref = res[d].get('ws_sparsity1.0', (np.array([]),))[0]
                diff = e.mean() - ref.mean() if len(ref) else np.nan
                line += ' {:>20s} {:>+8.2f}'.format('{:.2f} +- {:.2f} ({})'.format(
                    e.mean(), e.std(ddof=1) if len(e) > 1 else 0, len(e)), diff)
            else:
                line += ' {:>20s} {:>8s}'.format('-', '')
        print(line)
    print('\npaper, difference to the complete graph:')
    for g in ORDER[1:]:
        print('  paper {:<22s} {:+.2f}'.format(PAPER[g][0], PAPER[g][1] - PAPER['ws_sparsity1.0'][1]))

    if len(dirs) >= 2:
        from scipy import stats
        d0, d1 = dirs[0], dirs[1]
        print('\n{} vs {} (difference of means, Welch t-test over seeds)'.format(d1, d0))
        for g in ORDER:
            if g in res[d0] and g in res[d1] and len(res[d0][g][0]) > 1 and len(res[d1][g][0]) > 1:
                x, y = res[d0][g][0], res[d1][g][0]
                t, p = stats.ttest_ind(y, x, equal_var=False)
                print('  {:<24s} {:+.2f} points (p = {:.2f})'.format(PAPER[g][0], y.mean() - x.mean(), p))

    print('\nTraining time per epoch (s; median over the run, testing not included), mean over runs')
    for d in dirs:
        parts = []
        for g in ORDER:
            if g in res[d] and len(res[d][g][1]):
                parts.append('{} {:.1f}'.format(PAPER[g][0].split()[0] if g != 'mcvisualraw' else 'macaque-vis',
                                                res[d][g][1].mean()))
        allt = np.concatenate([res[d][g][1] for g in res[d] if len(res[d][g][1])]) if res[d] else np.array([])
        if len(allt):
            print('  {:<10s} {}   | all runs {:.1f} s/epoch, about {:.0f} min per 200-epoch run'.format(
                d, ', '.join(parts), allt.mean(), allt.mean() * 200 / 60))


if __name__ == '__main__':
    main()
