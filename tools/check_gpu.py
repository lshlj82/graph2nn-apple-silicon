#!/usr/bin/env python3
"""Check a GPU machine before running the paper's ImageNet experiments (CUDA or Apple-Silicon MPS).

    python tools/check_gpu.py                       # all ImageNet model families, baseline + best graph
    python tools/check_gpu.py --task cnn_imagenet resnet34_imagenet
    python tools/check_gpu.py --skip-models         # only the multi-GPU (DDP) test

1. devices      PyTorch version, GPUs and their memory
2. models       for each family: the paper's model (FLOPs matched as in the paper) trains a few steps on a
                synthetic batch at the paper's total batch size on one GPU; reports time per step, peak
                memory, and the projected GPU time of one 100-epoch ImageNet run (compute only: data
                loading must keep up, see tools/check_imagenet.py). If the batch does not fit, it tries
                half the batch (the per-GPU batch of a 2-GPU DDP run: paper_compare.py setup --num-gpus 2).
                Each test runs in its own process on the GPU with the most free memory (or --gpu N).
3. multi-GPU    with 2+ CUDA GPUs: the MLP trains three CIFAR-10 epochs with NUM_GPUS 2 (DDP) and with
                NUM_GPUS 1; both must finish, write one result, and reach similar test error.
                --cpu-ddp runs the same test with two CPU processes (gloo), e.g. on a Mac.
"""

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import torch                                            # noqa: E402

import train_net                                        # noqa: E402
from check_paper_models import TASKS, DEFAULTS, config_files   # noqa: E402
from pycls.config import cfg                            # noqa: E402
from pycls.models import model_builder                  # noqa: E402

IMAGENET_TRAIN = 1281167
_ok = {'pass': True}


def report(status, msg):
    print('  [{}] {}'.format(status, msg))
    if status == 'FAIL':
        _ok['pass'] = False


def gpu_usage():
    """[(index, name, used GB, total GB, utilization %)] from nvidia-smi, or None"""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
        rows = [[x.strip() for x in l.split(',')] for l in out.strip().splitlines()]
        return [(int(r[0]), r[1], float(r[2]) / 1024, float(r[3]) / 1024, float(r[4])) for r in rows]
    except Exception:
        return None


def devices():
    """prints the devices; returns (device type, GPU index to use)"""
    print('\n1. devices')
    print('  torch {}'.format(torch.__version__))
    if torch.cuda.is_available():
        use = gpu_usage()
        if not use:
            report('OK', '{} CUDA GPU(s)'.format(torch.cuda.device_count()))
            return 'cuda', 0
        for i, name, used, total, util in use:
            busy = used > 1 or util > 5
            report('WARNING' if busy else 'OK', 'cuda:{} {} ({:.0f} GB){}'.format(
                i, name, total, ', in use by other jobs: {:.1f} GB, {:.0f}% busy'.format(used, util) if busy else ', idle'))
        best = min(use, key=lambda r: r[2])
        if any(r[2] > 1 or r[4] > 5 for r in use):
            print('     other jobs share the GPUs: timings below are slower and memory tighter than on idle GPUs')
        print('     model tests use cuda:{} (most free memory)'.format(best[0]))
        return 'cuda', best[0]
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        report('OK', 'mps (Apple Silicon)')
        return 'mps', None
    report('WARNING', 'no GPU found; model timings run on the CPU')
    return 'cpu', None


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def one_test(cfg_file, batch, steps):
    """(run in a fresh process) train `steps` SGD steps on a synthetic batch; prints the result line"""
    cfg.merge_from_other_cfg(DEFAULTS)
    cfg.merge_from_file(cfg_file)
    cfg.NUM_GPUS = 1
    cfg.RGRAPH.SEED_TRAIN = cfg.RGRAPH.SEED_TRAIN_START
    cfg.OUT_DIR = '.'
    cfg.DEVICE = 'cpu'
    train_net._CFG_FILE = os.path.basename(cfg_file)
    with contextlib.redirect_stdout(io.StringIO()):
        train_net.match_complexity()
        model = model_builder.build_model()
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(0)
    model = model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.OPTIM.BASE_LR, momentum=0.9, nesterov=True)
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, cfg.MODEL.NUM_CLASSES, (batch,), device=device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    try:
        times = []
        for i in range(steps + 2):               # 2 warm-up steps
            sync(device)
            t = time.time()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sync(device)
            if i >= 2:
                times.append(time.time() - t)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('CHECK_GPU OOM')
            return
        raise
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else float('nan')
    print('CHECK_GPU {} {}'.format(sum(times) / len(times), peak))


def try_batch(gpu, cfg_file, batch, steps):
    """runs one_test in a fresh process (clean GPU memory); returns (s/step, peak GB), 'oom', or an error text"""
    env = dict(os.environ, PYTHONPATH=ROOT)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), '--one', cfg_file, str(batch), str(steps)],
                       capture_output=True, text=True, env=env)
    m = re.search(r'CHECK_GPU (\S+)(?: (\S+))?', p.stdout)
    if m and m.group(1) == 'OOM':
        return 'oom'
    if m:
        return float(m.group(1)), float(m.group(2))
    return (p.stdout + p.stderr)[-1500:]


def models(gpu, tasks, steps):
    print('\n2. models (synthetic batch, {} steps after warm-up, fp32 as in the paper)'.format(steps))
    for task in tasks:
        for f in config_files(task, False):
            cfg.merge_from_other_cfg(DEFAULTS)
            cfg.merge_from_file(f)
            batch = cfg.TRAIN.BATCH_SIZE
            name = '{} {}'.format(task, 'baseline' if f.endswith('_baseline.yaml') else 'best graph')
            for b in (batch, batch // 2):
                r = try_batch(gpu, f, b, steps)
                if r == 'oom':
                    report('WARNING', '{}: batch {} does not fit in memory{}'.format(
                        name, b, '' if b == batch else ' (needs 4+ GPUs of this size, or a larger GPU)'))
                    continue
                if isinstance(r, str):
                    report('FAIL', '{}: error:\n{}'.format(name, r))
                    break
                s, peak = r
                hours = IMAGENET_TRAIN / batch * 100 * s * (batch / b) / 3600
                report('OK', '{}: batch {}{}: {:.3f} s/step, peak {:.1f} GB -> ~{:.0f} GPU-h per 100-epoch run{}'.format(
                    name, b, '' if b == batch else ' (per GPU of a 2-GPU run)', s, peak, hours,
                    '' if b == batch else ' (~{:.0f} h on 2 GPUs)'.format(hours / 2)))
                break


def run_mlp(n, out, cpu):
    c = os.path.join(ROOT, 'configs/baselines/cifar10/mlp_cifar10/best/mlp_bs128_1gpu_layer3_best.yaml')
    cmd = [sys.executable, os.path.join(ROOT, 'tools/train_net.py'), '--cfg', c, 'OUT_DIR', out, 'NUM_GPUS', str(n),
           'OPTIM.MAX_EPOCH', '3', 'RGRAPH.SEED_TRAIN_START', '1', 'RGRAPH.SEED_TRAIN_END', '2',
           'DATA_LOADER.NUM_WORKERS', '2', 'LOG_DEST', 'stdout', 'PORT', str(12000 + os.getpid() % 1000)]
    if cpu:
        cmd += ['DEVICE', 'cpu']
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT))
    res = os.path.join(out, 'results_epoch3.txt')
    lines = open(res).read().strip().splitlines() if os.path.isfile(res) else []
    return p.returncode, lines, time.time() - t, p.stdout[-2000:] + p.stderr[-3000:]


def ddp(cpu):
    print('\n3. multi-GPU (DDP), MLP on CIFAR-10, 3 epochs')
    if not cpu and torch.cuda.device_count() < 2:
        report('OK', 'skipped: fewer than 2 CUDA GPUs (use --cpu-ddp to test the multi-process code on the CPU)')
        return
    data = os.path.join(ROOT, 'pycls/datasets/data/cifar10')
    if not os.path.exists(data):
        report('WARNING', 'skipped: CIFAR-10 not found in {}; download it as described in the README (Data), e.g.\n'
               '     mkdir -p pycls/datasets/data && ln -sfn ~/datasets.noindex/cifar-10-batches-py '
               'pycls/datasets/data/cifar10'.format(os.path.relpath(data, ROOT)))
        return
    with tempfile.TemporaryDirectory() as tmp:
        errs = {}
        for n in (1, 2):
            rc, lines, t, tail = run_mlp(n, os.path.join(tmp, 'n{}'.format(n)), cpu)
            if rc != 0 or len(lines) != 1:
                report('FAIL', 'NUM_GPUS {}: exit code {}, {} result lines; end of the log:\n{}'.format(n, rc, len(lines), tail))
                return
            errs[n] = float(lines[0].split(',')[3])
            report('OK', 'NUM_GPUS {}: test top-1 error {:.2f}% after 3 epochs ({:.0f} s)'.format(n, errs[n], t))
        d = abs(errs[1] - errs[2])
        report('OK' if d < 3 else 'WARNING', 'difference {:.2f} points (different batch order; expected within ~3)'.format(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', nargs='+', default=[t for t in TASKS if t != 'mlp_cifar10'],
                    choices=[t for t in TASKS if t != 'mlp_cifar10'])
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--skip-models', action='store_true')
    ap.add_argument('--skip-ddp', action='store_true')
    ap.add_argument('--cpu-ddp', action='store_true', help='test multi-process training with 2 CPU processes')
    ap.add_argument('--gpu', type=int, default=None, help='GPU for the model tests (default: most free memory)')
    ap.add_argument('--one', nargs=3, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        one_test(a.one[0], int(a.one[1]), int(a.one[2]))
        return
    kind, gpu = devices()
    if not a.skip_models:
        models(a.gpu if a.gpu is not None else gpu, a.task, a.steps)
    if not a.skip_ddp:
        ddp(a.cpu_ddp)
    print('\nRESULT: {}'.format('all checks passed' if _ok['pass'] else 'some checks FAILED'))
    sys.exit(0 if _ok['pass'] else 1)


if __name__ == '__main__':
    main()
