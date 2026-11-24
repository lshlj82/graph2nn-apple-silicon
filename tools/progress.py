#!/usr/bin/env python3
"""Progress and estimated finish time of a launch_parallel.sh experiment.

    python tools/progress.py g52                     # one line: done/total, running, rate, finish time
    python tools/progress.py g52 -v                  # also each run in progress
    python tools/progress.py g52 --hours 3           # rate from the last 3 hours
    python tools/progress.py g52 --seeds 3           # launched with EXTRA="RGRAPH.SEED_TRAIN_END 4"

Counts finished seeds in the results files (total = configs x seeds in configs/.../<division>; runs of
earlier launches count too, also without per-seed logs), shows the
training processes of this experiment that are running or paused (from the process list, so logs left by
earlier failed or interrupted attempts are ignored), and estimates the finish time from the rate at which runs
finished recently (finish time of a run = last change of its log), so pauses and other load on the machine are
taken into account. The estimate needs runs finishing over at least 45 minutes.
"""

import argparse
import datetime as dt
import glob
import os
import re
import subprocess
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('division')
    ap.add_argument('--task', default='mlp_cifar10')
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--hours', type=float, default=6.0, help='window for the finishing rate (default 6 h)')
    ap.add_argument('-v', '--verbose', action='store_true', help='list the runs in progress')
    ap.add_argument('--seeds', type=int, default=None,
                    help='seeds per config, if the launch overrode the configs (EXTRA="RGRAPH.SEED_TRAIN_END ...")')
    a = ap.parse_args()
    cfgs = sorted(glob.glob(os.path.join(ROOT, 'configs/baselines', a.dataset, a.task, a.division, '*.yaml')))
    total = 0
    for cf in cfgs:
        rg = yaml.safe_load(open(cf))['RGRAPH']
        total += a.seeds or int(rg.get('SEED_TRAIN_END', 2)) - int(rg.get('SEED_TRAIN_START', 1))
    out = os.path.join(ROOT, 'checkpoint', a.dataset, a.task, a.division)
    finished = []
    names = {os.path.splitext(os.path.basename(cf))[0] for cf in cfgs}
    for d in glob.glob(os.path.join(out, '*')):
        if os.path.basename(d) not in names:           # runs of configs not in this experiment (e.g. parked)
            continue
        epoch = 200
        cy = os.path.join(d, 'config.yaml')
        if os.path.isfile(cy):
            epoch = int((yaml.safe_load(open(cy)).get('OPTIM') or {}).get('MAX_EPOCH', 200))
        rf = os.path.join(d, 'results_epoch{}.txt'.format(epoch))
        if not os.path.isfile(rf):
            continue
        seeds = {int(l.split(',')[-1]) for l in open(rf) if l.strip()}
        for seed in seeds:
            if a.seeds and seed > a.seeds:             # seeds beyond the ones launched (e.g. from earlier runs)
                continue
            log = os.path.join(d, 'run_seed{}.log'.format(seed))
            # finish time = last change of the seed's log; runs without one count as done but not for the rate
            finished.append(os.path.getmtime(log) if os.path.isfile(log) else None)
    # training processes of this experiment (running, or stopped by trainpause: state T)
    running = []
    ps = subprocess.run(['ps', '-axo', 'stat=,command='], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        stat, _, cmd = line.strip().partition(' ')
        cmd = cmd.strip()
        m = re.match(r'\S*python[\d.]*\s+(?:\S*/)?tools/train_net\.py\s.*OUT_DIR (\S+)', cmd)
        if not m or os.path.basename(os.path.dirname(m.group(1).rstrip('/'))) != a.division:
            continue
        sm = re.search(r'RGRAPH\.SEED_TRAIN_START (\d+)', cmd)
        seed = sm.group(1) if sm else 'all'
        log = os.path.join(ROOT, m.group(1), 'run_seed{}.log'.format(seed))
        txt = open(log, errors='replace').read() if os.path.isfile(log) else ''
        ep = re.findall(r'"_type": "train_epoch", "epoch": "(\d+)/(\d+)"', txt)
        eta = re.findall(r'"_type": "train_epoch".*?"eta": "([^"]+)"', txt)
        running.append((os.path.basename(m.group(1).rstrip('/')).replace('mlp_bs128_1gpu_layer3_ws_', ''), seed,
                        '{}/{}'.format(*ep[-1]) if ep else 'starting', eta[-1] if eta else '-', stat.startswith('T')))
    n = len(finished)
    now = time.time()
    paused = sum(r[4] for r in running)
    recent = sorted(t for t in finished if t is not None and now - t < a.hours * 3600)
    if n >= total and total:
        times = [t for t in finished if t is not None]
        eta = 'all done' + (' (last finished {})'.format(dt.datetime.fromtimestamp(max(times)).strftime('%a %H:%M')) if times else '')
    elif len(recent) >= 2 and now - recent[0] >= 45 * 60:
        rate = (len(recent) - 1) / (now - recent[0]) * 3600      # runs finished since the first one in the window
        left = (total - n) / rate
        eta = '{:.1f} runs/h, ~{:.0f} h left, finish ~{}'.format(
            rate, left, (dt.datetime.now() + dt.timedelta(hours=left)).strftime('%a %d %b %H:%M'))
    else:
        eta = 'no estimate yet (needs runs finishing over ~45 min)'
    print('{}: {}/{} done ({:.0f}%) | {} running{} | {}'.format(
        a.division, n, total, 100 * n / max(total, 1), len(running) - paused,
        ', {} paused'.format(paused) if paused else '', eta))
    if a.verbose:
        for name, seed, ep, eta_run, stopped in sorted(running):
            print('  {} seed {}: epoch {}, eta {}{}'.format(name, seed, ep, eta_run, '  (paused)' if stopped else ''))


if __name__ == '__main__':
    main()
