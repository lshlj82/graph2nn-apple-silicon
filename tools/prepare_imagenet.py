#!/usr/bin/env python3
"""Prepare ImageNet (ILSVRC-2012) in the layout graph2nn expects:

    <dst>/train/<wnid>/*.JPEG
    <dst>/val/<wnid>/*.JPEG

Three sub-commands:

  kaggle    From the Kaggle "imagenet-object-localization-challenge" download.
            train/ is already split into class folders; val/ is flat and is
            sorted into class folders using LOC_val_solution.csv.

              python tools/prepare_imagenet.py kaggle \
                  --src /Volumes/SSD/imagenet-kaggle --dst /Volumes/SSD/imagenet --move

  official  From the image-net.org tarballs (ILSVRC2012_img_train.tar,
            ILSVRC2012_img_val.tar, ILSVRC2012_devkit_t12.tar.gz).

              python tools/prepare_imagenet.py official \
                  --train-tar ILSVRC2012_img_train.tar --val-tar ILSVRC2012_img_val.tar \
                  --devkit ILSVRC2012_devkit_t12.tar.gz --dst /Volumes/SSD/imagenet

  subset    Make a smaller ImageNet (e.g. ImageNet-100) out of a prepared one,
            using symlinks (no extra disk space).

              python tools/prepare_imagenet.py subset --src /Volumes/SSD/imagenet \
                  --dst /Volumes/SSD/imagenet100 --classes 100 --train-per-class 300

Afterwards link the result into the code base:

    mkdir -p pycls/datasets/data && ln -sfn /Volumes/SSD/imagenet pycls/datasets/data/imagenet
"""

import argparse
import csv
import os
import random
import shutil
import sys
import tarfile


def _extract(tar, dst):
    # Python >= 3.11.4 / 3.12 supports safe extraction filters
    if hasattr(tarfile, 'data_filter'):
        tar.extractall(dst, filter='data')
    else:
        tar.extractall(dst)


def _place(src, dst, move):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def _progress(i, n, what):
    if i % 2000 == 0 or i == n:
        sys.stdout.write('\r  {}: {}/{}'.format(what, i, n))
        sys.stdout.flush()
        if i == n:
            sys.stdout.write('\n')


def _find(root, name):
    for dirpath, dirnames, filenames in os.walk(root):
        if name in filenames or name in dirnames:
            return os.path.join(dirpath, name)
    return None


# ----------------------------------------------------------------------------
def cmd_kaggle(a):
    # Kaggle has both Annotations/CLS-LOC and Data/CLS-LOC; we need the images
    train_src = next((d for d, _, _ in os.walk(a.src)
                      if d.rstrip(os.sep).endswith(os.path.join('Data', 'CLS-LOC'))), None)
    csv_path = _find(a.src, 'LOC_val_solution.csv')
    assert train_src and csv_path, \
        'Could not find Data/CLS-LOC and LOC_val_solution.csv under {}'.format(a.src)
    os.makedirs(a.dst, exist_ok=True)

    # train: already one folder per class -> move/copy the class folders
    tr_src, tr_dst = os.path.join(train_src, 'train'), os.path.join(a.dst, 'train')
    classes = sorted(d for d in os.listdir(tr_src) if not d.startswith('.'))
    print('train: {} classes'.format(len(classes)))
    os.makedirs(tr_dst, exist_ok=True)
    for i, c in enumerate(classes, 1):
        s, d = os.path.join(tr_src, c), os.path.join(tr_dst, c)
        if not os.path.exists(d):
            shutil.move(s, d) if a.move else shutil.copytree(s, d)
        _progress(i, len(classes), 'train classes')

    # val: flat folder + CSV "ImageId,PredictionString" (first token = wnid)
    va_src = os.path.join(train_src, 'val')
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print('val: {} images'.format(len(rows)))
    for i, r in enumerate(rows, 1):
        wnid = r['PredictionString'].split()[0]
        name = r['ImageId'] + '.JPEG'
        _place(os.path.join(va_src, name), os.path.join(a.dst, 'val', wnid, name), a.move)
        _progress(i, len(rows), 'val images')
    verify(a.dst)


# ----------------------------------------------------------------------------
def cmd_official(a):
    import scipy.io
    os.makedirs(a.dst, exist_ok=True)
    tmp = os.path.join(a.dst, '_tmp')
    os.makedirs(tmp, exist_ok=True)

    # devkit: ILSVRC2012_ID -> wnid, and the val ground truth (one ID per image)
    with tarfile.open(a.devkit) as t:
        _extract(t, tmp)
    meta = scipy.io.loadmat(_find(tmp, 'meta.mat'), squeeze_me=True)['synsets']
    id2wnid = {int(s['ILSVRC2012_ID']): str(s['WNID']) for s in meta if int(s['ILSVRC2012_ID']) <= 1000}
    with open(_find(tmp, 'ILSVRC2012_validation_ground_truth.txt')) as f:
        val_ids = [int(l) for l in f if l.strip()]

    # train: outer tar holds one nXXXXXXXX.tar per class
    print('extracting train (this takes a while)...')
    with tarfile.open(a.train_tar) as outer:
        members = [m for m in outer.getmembers() if m.name.endswith('.tar')]
        for i, m in enumerate(members, 1):
            wnid = os.path.basename(m.name)[:-4]
            d = os.path.join(a.dst, 'train', wnid)
            if not os.path.isdir(d) or not os.listdir(d):
                os.makedirs(d, exist_ok=True)
                with tarfile.open(fileobj=outer.extractfile(m)) as inner:
                    _extract(inner, d)
            _progress(i, len(members), 'train classes')

    # val: flat tar, images sorted by name match the ground-truth order
    print('extracting val...')
    vtmp = os.path.join(tmp, 'val')
    with tarfile.open(a.val_tar) as t:
        _extract(t, vtmp)
    names = sorted(n for n in os.listdir(vtmp) if n.endswith('.JPEG'))
    assert len(names) == len(val_ids), 'val images ({}) != labels ({})'.format(len(names), len(val_ids))
    for i, (n, cid) in enumerate(zip(names, val_ids), 1):
        _place(os.path.join(vtmp, n), os.path.join(a.dst, 'val', id2wnid[cid], n), move=True)
        _progress(i, len(names), 'val images')
    shutil.rmtree(tmp, ignore_errors=True)
    verify(a.dst)


# ----------------------------------------------------------------------------
def cmd_subset(a):
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    classes = sorted(d for d in os.listdir(os.path.join(src, 'train')) if not d.startswith('.'))
    rng = random.Random(a.seed)
    if a.classes < len(classes):
        classes = sorted(rng.sample(classes, a.classes))
    for split, limit in (('train', a.train_per_class), ('val', None)):
        for i, c in enumerate(classes, 1):
            s = os.path.join(src, split, c)
            d = os.path.join(dst, split, c)
            os.makedirs(d, exist_ok=True)
            ims = sorted(n for n in os.listdir(s) if not n.startswith('.'))
            if limit and limit < len(ims):
                ims = sorted(rng.sample(ims, limit))
            for n in ims:
                link = os.path.join(d, n)
                if not os.path.lexists(link):
                    os.symlink(os.path.join(s, n), link)
            _progress(i, len(classes), split + ' classes')
    with open(os.path.join(dst, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes) + '\n')
    print('NOTE: labels will be 0..{}. Keeping the default MODEL.NUM_CLASSES 1000 keeps the\n'
          '      FLOPs-matching targets in train_net.py exact (unused outputs are harmless);\n'
          '      MODEL.NUM_CLASSES {} also works but shifts the matched FLOPs.'.format(len(classes) - 1, len(classes)))
    verify(dst)


# ----------------------------------------------------------------------------
def verify(root):
    """Print image/class counts and check train and val have the same classes."""
    counts = {}
    for split in ('train', 'val'):
        p = os.path.join(root, split)
        cls = sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))
        n = sum(len([f for f in os.listdir(os.path.join(p, c)) if not f.startswith('.')]) for c in cls)
        counts[split] = (cls, n)
        print('{:5s}: {:4d} classes, {:8d} images'.format(split, len(cls), n))
    if counts['train'][0] != counts['val'][0]:
        print('WARNING: train and val class folders differ!')
    else:
        print('OK: train and val have identical class folders')
    full = (len(counts['train'][0]) == 1000)
    if full and (counts['train'][1] != 1281167 or counts['val'][1] != 50000):
        print('WARNING: expected 1,281,167 train / 50,000 val images for full ImageNet-1k')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('kaggle'); k.add_argument('--src', required=True); k.add_argument('--dst', required=True)
    k.add_argument('--move', action='store_true', help='move files instead of copying (saves ~150 GB)')
    o = sub.add_parser('official')
    for x in ('--train-tar', '--val-tar', '--devkit', '--dst'):
        o.add_argument(x, required=True)
    s = sub.add_parser('subset'); s.add_argument('--src', required=True); s.add_argument('--dst', required=True)
    s.add_argument('--classes', type=int, default=100)
    s.add_argument('--train-per-class', type=int, default=None, help='default: all (~1300)')
    s.add_argument('--seed', type=int, default=0)
    v = sub.add_parser('verify'); v.add_argument('root')
    a = p.parse_args()
    {'kaggle': cmd_kaggle, 'official': cmd_official, 'subset': cmd_subset,
     'verify': lambda a: verify(a.root)}[a.cmd](a)


if __name__ == '__main__':
    main()
