#!/usr/bin/env python3
"""Compare your results with the original paper's raw results, for any of the paper's experiments.

The paper's raw per-seed results ship with the repo (analysis/final_results/). This script sets up the
paper's runs and compares your results with them, graph by graph, on the graph sets used for the paper's
cross-task comparisons (52 graphs with 64 nodes; 48 graphs with 16 nodes for EfficientNet).

Tasks: mlp_cifar10 (default), cnn_imagenet (8-layer CNN), resnet34_imagenet, resnet34sep_imagenet,
resnet50_imagenet, efficient_imagenet.

MLP on CIFAR-10 (52 graphs, ~1 day on one Apple-Silicon Mac, a few hours on an RTX 5090 server):

     python tools/paper_compare.py setup                     # all 52 graphs
     python tools/paper_compare.py setup --graphs 16         # or 16 graphs, reusing finished ones
     bash launch_parallel.sh mlp_cifar10 g52 1 3 0
     python tools/paper_compare.py compare

ImageNet (paper's hyperparameters: 100 epochs, total batch 256 (EfficientNet 512); baseline and best graph,
plus --graphs N graphs spread over the paper's error range; 3 seeds as in the paper):

     python tools/paper_compare.py --task cnn_imagenet setup --graphs 0 --seeds 3
     GPUS="0 1" bash launch_parallel.sh cnn_imagenet repro 8 2 8
     python tools/paper_compare.py --task cnn_imagenet compare

The ImageNet configs are the paper's 8-GPU configs, run in one process per job with the same total batch
(--num-gpus 1, default); --num-gpus 2 uses PyTorch DDP over two GPUs instead (one job at a time, no GPUS).
`compare` works with partial results and writes checkpoint/paper_compare_<task>.pdf.
"""

import argparse
import glob
import os
import re
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_fig(fig, path, dpi=130):
    """write the figure as PDF (for LaTeX documents) and as PNG next to it (for the README and quick viewing)"""
    base = os.path.splitext(path)[0]
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=dpi, bbox_inches='tight')
ANALYSIS = os.path.join(ROOT, 'analysis')

# task: dataset, config prefix of the paper's runs, raw-results name, epochs, graph set, run-set division, title
TASKS = {
    'mlp_cifar10': ('cifar10', 'mlp_bs128_1gpu_layer3', 'mlp_cifar10', 200, 'graphs_n64_52', 'g52',
                    '5-layer MLP on CIFAR-10'),
    'cnn_imagenet': ('imagenet', 'cnn6_bs256_8gpu_64d', 'cnn6_imagenet', 100, 'graphs_n64_52', 'repro',
                     '8-layer CNN on ImageNet'),
    'resnet34_imagenet': ('imagenet', 'R-34_bs256_8gpu', 'resnet34_imagenet', 100, 'graphs_n64_52', 'repro',
                          'ResNet-34 on ImageNet'),
    'resnet34sep_imagenet': ('imagenet', 'R-34_bs256_8gpu', 'resnet34_sep_imagenet', 100, 'graphs_n64_52', 'repro',
                             'ResNet-34-sep on ImageNet'),
    'resnet50_imagenet': ('imagenet', 'R-50_bs256_8gpu', 'resnet50_imagenet', 100, 'graphs_n64_52', 'repro',
                          'ResNet-50 on ImageNet'),
    'efficient_imagenet': ('imagenet', 'EN-B0_bs512_8gpu_nms', 'efficientnet_imagenet', 100, 'graphs_n16_48', 'repro',
                           'EfficientNet-B0 on ImageNet'),
}
BASE = ('baseline',)
_PAT = re.compile(r'trans(\w+?)_talkmode.*?_sparsity([\d.]+)_p([\d.]+)_agg\w+?_graphseed(\d+)_')


class T:
    """settings of the selected task (filled in by use_task)"""


def use_task(task):
    ds, prefix, raw, epoch, graphs, division, title = TASKS[task]
    T.task, T.dataset, T.prefix, T.raw, T.epoch, T.graphs, T.division, T.title = \
        task, ds, prefix, raw, epoch, graphs, division, title
    T.cfg_root = os.path.join(ROOT, 'configs/baselines', ds, task)
    T.cfg_div = os.path.join(T.cfg_root, division)
    T.ckpt = os.path.join(ROOT, 'checkpoint', ds, task)


def parse(name):
    """config/dir name -> (transform, sparsity, p, graphseed) or None"""
    m = _PAT.search(name)
    if not m:
        return None
    return m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))


def is_base(k):
    return k[0] == 'linear' or k[1] >= 0.999


def graph_set():
    g = np.load(os.path.join(ANALYSIS, T.graphs + '.npy'))   # n, sparsity, p, seed, C, L
    return g[g[:, 1] < 1]


def parse_dir(d):
    """Like parse(), but falls back to the run's saved config.yaml (needed for
    the `best` division, whose names are just ..._baseline / ..._best)."""
    k = parse(os.path.basename(d))
    if k:
        return k
    f = os.path.join(d, 'config.yaml')
    if not os.path.isfile(f):
        return None
    import yaml
    c = yaml.safe_load(open(f))
    trans = c['RESNET']['TRANS_FUN'].replace('_transform', '')
    return trans, float(c['RGRAPH']['SPARSITY']), float(c['RGRAPH']['P']), int(c['RGRAPH']['SEED_GRAPH'])


def cfg_key(f):
    import yaml
    c = yaml.safe_load(open(f))
    return (c['RESNET']['TRANS_FUN'].replace('_transform', ''), float(c['RGRAPH']['SPARSITY']),
            float(c['RGRAPH']['P']), int(c['RGRAPH']['SEED_GRAPH']))


def match(g, sparsity, p, seed):
    return abs(g[1] - sparsity) < 2e-6 and abs(g[2] - p) < 2e-6 and int(round(g[3])) == seed


def best_key():
    f = [f for f in glob.glob(os.path.join(T.cfg_root, 'best', '*_best.yaml')) if T.prefix in os.path.basename(f)][0]
    return cfg_key(f)[1:]


# ----------------------------------------------------------------------------
def choose(n, chosen_all):
    """indices of n graphs: graphs with >= 2 finished seeds first, the rest spread over the paper's error range"""
    G = graph_set()
    if n >= len(G):
        return list(range(len(G)))
    paper, mine = load_paper(), load_mine()
    pm = np.array([lookup(paper, g).mean() for g in G])
    done = np.array([len(lookup(mine, g)) if lookup(mine, g) is not None else 0 for g in G])
    keep = [i for i in np.argsort(-done) if done[i] >= 2][:n]
    rest = [i for i in np.argsort(pm) if i not in keep]
    need = n - len(keep)
    if need > 0:
        keep += [rest[j] for j in np.unique(np.round(np.linspace(0, len(rest) - 1, need)).astype(int))]
    return keep


def setup_mlp(a):
    """Create the 52-graph run set, or (--graphs N) a subset of N graphs; unused configs are parked in
    g52_unused/ (run `setup` without --graphs to restore all 52)."""
    os.makedirs(T.cfg_div, exist_ok=True)
    parked = T.cfg_div + '_unused'
    if os.path.isdir(parked):
        for f in os.listdir(parked):
            shutil.move(os.path.join(parked, f), os.path.join(T.cfg_div, f))
        os.rmdir(parked)
    cfgs = glob.glob(os.path.join(T.cfg_root, 'all', '*.yaml'))
    chosen = {}
    G = graph_set()
    for g in G:
        hit = [f for f in cfgs if (lambda k: k and k[0] == 'talklinear' and match(g, *k[1:]))(parse(os.path.basename(f)))]
        assert len(hit) == 1, 'graph {} matched {} configs'.format(g[:4], len(hit))
        dst = os.path.join(T.cfg_div, os.path.basename(hit[0]))
        if not os.path.exists(dst):
            shutil.copy(hit[0], dst)
        chosen[tuple(g[:4])] = dst
    n = len(G) if a.graphs is None else a.graphs
    if n < len(G):
        keep = choose(n, chosen)
        os.makedirs(parked, exist_ok=True)
        for i, g in enumerate(G):
            if i not in keep:
                f = chosen[tuple(g[:4])]
                shutil.move(f, os.path.join(parked, os.path.basename(f)))
        print('kept {} graphs; {} parked in {}'.format(len(keep), len(G) - len(keep), os.path.relpath(parked, ROOT)))
    print('{} configs in {}'.format(len(glob.glob(os.path.join(T.cfg_div, '*.yaml'))), os.path.relpath(T.cfg_div, ROOT)))
    print('next: bash launch_parallel.sh mlp_cifar10 g52 1 <JOBS> 0')


def setup_imagenet(a):
    """Baseline + best graph (the paper's `best` configs) + N graphs of the graph set, as the paper's 8-GPU
    configs with NUM_GPUS set to --num-gpus (same total batch) and --seeds training seeds."""
    import yaml
    if os.path.isdir(T.cfg_div):
        shutil.rmtree(T.cfg_div)
    os.makedirs(T.cfg_div)
    src = [f for f in glob.glob(os.path.join(T.cfg_root, 'best', '*.yaml')) if T.prefix in os.path.basename(f)]
    G = graph_set()
    cfgs = [f for f in glob.glob(os.path.join(T.cfg_root, 'all', '*.yaml')) if T.prefix in os.path.basename(f)]
    bk = best_key()
    picks = [i for i in choose(a.graphs + 1, None) if not match(G[i], *bk)][:a.graphs] if a.graphs else []
    for i in picks:
        hit = [f for f in cfgs if (lambda k: k and match(G[i], *k[1:]))(parse(os.path.basename(f)))]
        assert len(hit) == 1, 'graph {} matched {} configs'.format(G[i][:4], len(hit))
        src.append(hit[0])
    for f in src:
        c = yaml.safe_load(open(f))
        c['NUM_GPUS'] = a.num_gpus
        c['RGRAPH']['SEED_TRAIN_START'] = 1
        c['RGRAPH']['SEED_TRAIN_END'] = 1 + a.seeds
        with open(os.path.join(T.cfg_div, os.path.basename(f)), 'w') as out:
            yaml.safe_dump(c, out, default_flow_style=False)
    print('{} configs x {} seeds in {} (total batch {}, NUM_GPUS {})'.format(
        len(src), a.seeds, os.path.relpath(T.cfg_div, ROOT), c['TRAIN']['BATCH_SIZE'], a.num_gpus))
    if a.num_gpus == 1:
        print('next (e.g. two GPUs, one job each, 8 loader workers per job):\n'
              '  GPUS="0 1" bash launch_parallel.sh {} {} 8 2 8'.format(T.task, T.division))
    else:
        print('next (one DDP job over {} GPUs at a time):\n'
              '  bash launch_parallel.sh {} {} 8 1 8'.format(a.num_gpus, T.task, T.division))


def setup(a):
    (setup_mlp if T.dataset == 'cifar10' else setup_imagenet)(a)


# ----------------------------------------------------------------------------
def load_mine():
    """{key: array of final top-1 errors over seeds}; key = BASE or (sparsity, p, seed)"""
    out = {}
    for d in glob.glob(os.path.join(T.ckpt, '*', '*')):
        if 'keepFalse' in os.path.basename(d):
            continue                                              # per-layer graphs (RGRAPH.KEEP_GRAPH False): not the paper's setup
        f = os.path.join(d, 'results_epoch{}.txt'.format(T.epoch))
        if not os.path.isfile(f):
            continue
        k = parse_dir(d)
        if not k:
            continue
        r = np.loadtxt(f, delimiter=',', ndmin=2)
        per_seed = {int(row[7]): row[3] for row in r}       # one row per seed (last one wins)
        key = BASE if is_base(k) else k[1:]
        out.setdefault(key, {}).update(per_seed)
    return {k: np.array(list(v.values())) for k, v in out.items()}


def load_paper():
    """{key: array of the paper's final top-1 errors over seeds}"""
    if T.dataset == 'cifar10':
        d = os.path.join(ANALYSIS, 'final_results', T.raw)
        pairs = [(os.path.join(d, 'names_{}_epoch{}.npy'.format(T.raw, T.epoch)),
                  os.path.join(d, 'results_{}_epoch{}.npy'.format(T.raw, T.epoch)))]
    else:
        pairs = []
        for s in (1, 2, 3):
            d = os.path.join(ANALYSIS, 'final_results', '{}_seed{}'.format(T.raw, s))
            pairs.append((glob.glob(os.path.join(d, 'names_*epoch{}.npy'.format(T.epoch)))[0],
                          glob.glob(os.path.join(d, 'results_*epoch{}.npy'.format(T.epoch)))[0]))
    out = {}
    for nf, rf in pairs:
        for row, r in zip(np.load(nf, allow_pickle=True), np.load(rf, allow_pickle=True)):
            r = np.asarray(r, dtype=float).reshape(-1, 8)
            k = (str(row[0]), float(row[4]), float(row[5]), int(float(row[7]))) if row[0] != 'linear' else ('linear', 1., 0., 0)
            key = BASE if is_base(k) else k[1:]
            out.setdefault(key, []).extend(r[:, 3])
    return {k: np.array(v) for k, v in out.items()}


def lookup(d, g):
    for k, v in d.items():
        if len(k) == 3 and match(g, *k):
            return v
    return None


def compare(a):
    from scipy import stats
    mine, paper = load_mine(), load_paper()
    G = graph_set()

    rows = []   # C, L, mine_mean, mine_std, n_seeds, paper_mean, paper_std
    for g in G:
        m, p = lookup(mine, g), lookup(paper, g)
        if m is not None and len(m) and p is not None:
            rows.append([g[4], g[5], m.mean(), m.std(), len(m), p.mean(), p.std()])
    rows = np.array(rows)

    def fmt(x):
        return '{:.2f} ± {:.2f} (n={})'.format(x.mean(), x.std(), len(x)) if x is not None and len(x) else '-'

    bk = best_key()
    mb = next((v for k, v in mine.items() if len(k) == 3 and np.allclose(k, bk, atol=2e-6)), None)
    pb = next((v for k, v in paper.items() if len(k) == 3 and np.allclose(k, bk, atol=2e-6)), None)
    print('\n=== {}, final top-1 error (%) ==='.format(T.title))
    print('{:<28s} {:>22s} {:>22s}'.format('', 'yours', 'paper (raw results)'))
    print('{:<28s} {:>22s} {:>22s}'.format('baseline (complete graph)', fmt(mine.get(BASE)), fmt(paper.get(BASE))))
    print('{:<28s} {:>22s} {:>22s}'.format('best graph (paper\'s pick)', fmt(mb), fmt(pb)))
    if mine.get(BASE) is not None and mb is not None:
        print('{:<28s} {:>22s} {:>22s}'.format('gain of best over baseline', '{:+.2f}'.format(mine[BASE].mean() - mb.mean()),
                                               '{:+.2f}'.format(paper[BASE].mean() - pb.mean())))

    if len(rows) < 3:
        print('\nGraph-by-graph comparison needs results for at least 3 graphs of the set '
              '(setup --graphs N; {} so far).'.format(len(rows)))
        return
    C, L, mm, ms, ns, pm, ps = rows.T
    base_m, base_p = mine.get(BASE), paper.get(BASE)
    r, rp = stats.pearsonr(mm, pm)
    rho, _ = stats.spearmanr(mm, pm)
    print('\n=== graph-by-graph comparison ({} graphs with results, {}-{} seeds each) ==='.format(
        len(rows), int(ns.min()), int(ns.max())))
    print('per-graph correlation, yours vs paper : Pearson r = {:.3f} (p = {:.1e}), Spearman = {:.3f}'.format(r, rp, rho))
    # what agreement to expect: the paper's own seeds, k (= your seeds) vs the rest, on the same graphs
    k_seeds = int(np.median(ns))
    E = [lookup(paper, g) for g in G if lookup(mine, g) is not None and len(lookup(mine, g))]
    n_paper = min(len(e) for e in E)
    E = np.array([e[:n_paper] for e in E])
    k = min(8, len(rows) // 2)
    noise = None
    if 1 <= k_seeds < n_paper and len(E) >= 4:
        rng = np.random.default_rng(0); rs, ov, dc, dl = [], [], [], []
        for _ in range(500):
            perm = np.array([rng.permutation(E.shape[1]) for _ in E])
            x = np.array([e[pp[:k_seeds]].mean() for e, pp in zip(E, perm)])
            y = np.array([e[pp[k_seeds:]].mean() for e, pp in zip(E, perm)])
            rs.append(stats.pearsonr(x, y)[0])
            tx, ty = list(np.argsort(x)[:k]), list(np.argsort(y)[:k])
            ov.append(len(set(tx) & set(ty)))
            dc.append(abs(C[tx].mean() - C[ty].mean())); dl.append(abs(L[tx].mean() - L[ty].mean()))
        noise = (np.median(ov), np.percentile(ov, 10), np.percentile(ov, 90), np.percentile(dc, 90), np.percentile(dl, 90))
        print('expected r from seed noise alone      : ~{:.2f} (10th-90th pct {:.2f}-{:.2f}; paper seeds {} vs {} on these graphs)'.format(
            np.median(rs), np.percentile(rs, 10), np.percentile(rs, 90), k_seeds, E.shape[1] - k_seeds))
    print('mean offset (yours - paper)          : {:+.2f} pts,  RMS diff {:.2f} pts'.format(
        (mm - pm).mean(), np.sqrt(((mm - pm) ** 2).mean())))
    if base_m is not None:
        print('graphs better than baseline          : yours {}/{}   paper {}/{}'.format(
            (mm < base_m.mean()).sum(), len(rows), (pm < base_p.mean()).sum(), len(rows)))
    tm, tp = set(np.argsort(mm)[:k]), set(np.argsort(pm)[:k])
    print('top-{} graphs shared with paper       : {}/{}{}'.format(k, len(tm & tp), k, '' if noise is None else
          '   (seed noise alone: ~{:.0f}, 10th-90th pct {:.0f}-{:.0f})'.format(*noise[:3])))
    print('sweet spot (mean C, L of top-{})      : yours C={:.2f} L={:.2f}   paper C={:.2f} L={:.2f}{}'.format(
        k, C[list(tm)].mean(), L[list(tm)].mean(), C[list(tp)].mean(), L[list(tp)].mean(), '' if noise is None else
        '   (seed noise alone: |dC| < {:.2f}, |dL| < {:.2f} in 90%)'.format(*noise[3:])))

    # ---- figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    lo, hi = min(mm.min(), pm.min()) - .1, max(mm.max(), pm.max()) + .1
    ax[0].errorbar(pm, mm, xerr=rows[:, 6], yerr=ms, fmt='o', ms=4, alpha=.7, capsize=0)
    ax[0].plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax[0].set_xlabel('paper top-1 error (%)'); ax[0].set_ylabel('your top-1 error (%)')
    ax[0].set_title('per graph: r = {:.2f}'.format(r))
    vmin, vmax = np.percentile(np.r_[mm, pm], [2, 98])
    for i, (vals, t) in enumerate([(pm, 'paper'), (mm, 'yours')], 1):
        sc = ax[i].scatter(C, L, c=vals, cmap='viridis_r', vmin=vmin, vmax=vmax, s=70, edgecolors='k', lw=.3)
        ax[i].set_xlabel('clustering C'); ax[i].set_ylabel('path length L'); ax[i].set_title(t)
        top = np.argsort(vals)[:k]
        ax[i].scatter(C[top], L[top], s=170, facecolors='none', edgecolors='r', lw=1.5, label='top {}'.format(k))
        ax[i].legend(loc='upper right', fontsize=8)
    fig.colorbar(sc, ax=ax[1:], label='top-1 error (%)')
    fig.suptitle('{}, {} relational graphs: reproduction vs. You et al. (2020)'.format(T.title, len(rows)))
    out = os.path.join(ROOT, 'checkpoint', 'paper_compare_{}.pdf'.format(T.task))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_fig(fig, out)
    print('\nfigure: {}'.format(os.path.relpath(out, ROOT)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', default='mlp_cifar10', choices=list(TASKS))
    s = p.add_subparsers(dest='cmd', required=True)
    sp = s.add_parser('setup')
    sp.add_argument('--graphs', type=int, default=None,
                    help='MLP: keep only N of the 52 graphs; ImageNet: N graphs besides baseline and best (default 0)')
    sp.add_argument('--seeds', type=int, default=3, help='ImageNet: training seeds (paper: 3)')
    sp.add_argument('--num-gpus', type=int, default=1, help='ImageNet: GPUs per job (1 = one process, full batch)')
    s.add_parser('compare')
    a = p.parse_args()
    use_task(a.task)
    {'setup': setup, 'compare': compare}[a.cmd](a)


if __name__ == '__main__':
    main()
