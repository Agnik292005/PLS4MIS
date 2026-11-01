"""
cmf_cache_dataset.py

Reads the NumPy cache written by data_prep/make_cmf_cache.py (PLAN.md section 3:
normalized float32 images, integer labels, statistics, geometry; memory-mapped
NumPy storage). Only the crop that is needed is read from disk.

Layout:  cache_dir/<train|val>/<case>_img.npy   float32, whole-volume z-scored
                              /<case>_lab.npy   uint8
                              /<case>_meta.json mean, std, shape, spacing, ..., present

Train item: one case row; the task is chosen by cmf_task_select.pick_task from
the classes present in the full label, as a function of (seed, case, epoch).
Because the epoch travels with the index (EpochShuffleSampler), worker
processes can stay alive between epochs.
"""
import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataloader.cmf_task_select import pick_task


def open_array(stem):
    """Open stem + '.b2nd' (chunked, lazy) or stem + '.npy' (memory-mapped).
    Slicing the result reads only the requested region."""
    if os.path.exists(stem + '.b2nd'):
        import blosc2
        return blosc2.open(stem + '.b2nd', mode='r')
    return np.load(stem + '.npy', mmap_mode='r')


def foreground_centre(lab, cls, rng):
    """A random voxel (z, y, x) whose label is `cls`, or None if the class does not occur."""
    arr = np.asarray(lab[...])
    idx = np.flatnonzero(arr == cls)
    if idx.size == 0:
        return None
    return np.unravel_index(int(idx[rng.randrange(idx.size)]), arr.shape)


def random_crop_padded(img, lab, patch_zyx, centre=None):
    """Random crop of size patch_zyx from memory-mapped arrays. A short axis is
    padded with 0 (image mean after z-score; label background), split evenly,
    exactly like cmf_transforms.PadToPatch. With `centre` (z, y, x) the crop
    is placed so that this voxel is inside it (as central as the borders allow)."""
    out_i = np.zeros(patch_zyx, dtype=np.float32)
    out_l = np.zeros(patch_zyx, dtype=np.uint8)
    src, dst = [], []
    for d, (dim, p) in enumerate(zip(img.shape, patch_zyx)):
        if dim >= p:
            if centre is None:
                s = random.randint(0, dim - p)
            else:
                s = min(max(int(centre[d]) - p // 2, 0), dim - p)
            src.append(slice(s, s + p))
            dst.append(slice(0, p))
        else:
            before = (p - dim) // 2
            src.append(slice(0, dim))
            dst.append(slice(before, before + dim))
    src, dst = tuple(src), tuple(dst)
    out_i[dst] = img[src]
    out_l[dst] = lab[src]
    return out_i, out_l


class EpochShuffleSampler(Sampler):
    """Yields (index, epoch) in a shuffled order that depends on (seed, epoch)."""

    def __init__(self, n, seed=42):
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + self.epoch)
        for i in torch.randperm(self.n, generator=g).tolist():
            yield (i, self.epoch)


class CMFCacheDataset(Dataset):
    def __init__(self, cache_dir, mode='train', patch_size=(192, 192, 192), num_classes=7, seed=42, fg_prob=0.0):
        if mode not in ('train', 'val'):
            raise ValueError("Unsupported mode: %s" % mode)
        self.mode, self.num_classes, self.seed = mode, num_classes, seed
        self.fg_prob = float(fg_prob)   # share of training crops centred on a labelled structure (0 = all random)
        self.patch_zyx = tuple(int(p) for p in patch_size)[::-1]  # given as x y z
        folder = os.path.join(cache_dir, mode)
        metas = sorted(glob.glob(os.path.join(folder, '*_meta.json')))
        if not metas:
            raise FileNotFoundError("no cache in %s (run data_prep/make_cmf_cache.py)" % folder)
        self.rows = []
        for m in metas:
            base = m[:-len('_meta.json')]
            with open(m) as f:
                meta = json.load(f)
            self.rows.append({'name': os.path.basename(base), 'img': base + '_img',
                              'lab': base + '_lab', 'present': meta['present']})
        print("CMFCacheDataset[%s]: %d cases from %s" % (mode, len(self.rows), folder))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        row = self.rows[index]
        img = open_array(row['img'])
        lab = open_array(row['lab'])
        if self.mode == 'val':
            return {'img_name': row['name'],
                    'image': torch.from_numpy(np.asarray(img[...], dtype=np.float32))[None],
                    'label': torch.from_numpy(np.asarray(lab[...]).astype(np.float32))}
        present = [c for c in row['present'] if 0 < c < self.num_classes]
        if not present:
            raise ValueError("case %s has no foreground label" % row['name'])
        task = pick_task(present, row['name'], epoch, self.seed)
        centre = None
        if self.fg_prob > 0:
            rng = random.Random('%d-%s-%d' % (self.seed, row['name'], epoch))   # same sample + epoch + seed: same decision
            if rng.random() < self.fg_prob:
                centre = foreground_centre(lab, task, rng)
        ci, cl = random_crop_padded(img, lab, self.patch_zyx, centre)
        for _ in range(9):  # like the stock TrainerCrop: avoid a constant crop
            if ci.max() != ci.min() or centre is not None:
                break
            ci, cl = random_crop_padded(img, lab, self.patch_zyx)
        return {'img_name': row['name'],
                'image': torch.from_numpy(ci)[None],
                'label': torch.from_numpy(cl.astype(np.float32)),
                'forced_class': torch.tensor(task)}
