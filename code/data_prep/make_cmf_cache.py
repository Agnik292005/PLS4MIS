"""
make_cmf_cache.py

One-time conversion of an nnU-Net style folder to the NumPy cache read by
dataloader/cmf_cache_dataset.py. Resumable: finished cases are skipped.

  python data_prep/make_cmf_cache.py --data_dir <folder with imagesTr/labelsTr/imagesVal/labelsVal> \
        --cache_dir <output folder> --workers 8
"""
import argparse
import json
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk

CHUNK = 64  # b2nd chunk edge: a 192^3 crop touches only the chunks it overlaps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.cmf_case_dataset import label_path_for


def case_name(image_path):
    f = os.path.basename(image_path)
    for suf in ('_0000.nii.gz', '.nii.gz'):
        if f.endswith(suf):
            return f[:-len(suf)]
    return f


def save_array(stem, arr, fmt):
    """Write stem + '.npy' or stem + '.b2nd' atomically (temp file, then rename)."""
    final = stem + '.' + fmt
    tmp = stem + '.tmp'
    if fmt == 'npy':
        with open(tmp, 'wb') as f:
            np.save(f, arr)
    else:
        import blosc2
        chunks = tuple(min(CHUNK, d) for d in arr.shape)
        blosc2.asarray(arr, urlpath=tmp, chunks=chunks, mode='w')
    os.replace(tmp, final)


def convert(job):
    image_path, label_path, out_dir, num_classes, fmt = job
    base = os.path.join(out_dir, case_name(image_path))
    if os.path.exists(base + '_meta.json'):
        return base, 'skipped'
    im = sitk.ReadImage(image_path)
    lb = sitk.ReadImage(label_path)
    if im.GetSize() != lb.GetSize():
        raise ValueError("size mismatch image %s vs label %s: %s" % (im.GetSize(), lb.GetSize(), image_path))
    img = sitk.GetArrayFromImage(im).astype(np.float32)
    lab = sitk.GetArrayFromImage(lb)
    if lab.min() < 0 or lab.max() >= num_classes:
        raise ValueError("label values outside 0..%d in %s: %s" % (num_classes - 1, label_path, np.unique(lab)))
    mean = float(img.mean(dtype=np.float64))
    std = float(img.std(dtype=np.float64))
    if not std > 1e-6:
        raise ValueError("constant image: %s" % image_path)
    img = ((img - mean) / std).astype(np.float32)
    present = [int(c) for c in np.unique(lab) if c != 0]
    save_array(base + '_img', img, fmt)
    save_array(base + '_lab', lab.astype(np.uint8), fmt)
    meta = {'source_image': image_path, 'source_label': label_path, 'mean': mean, 'std': std,
            'shape_zyx': list(img.shape), 'spacing_xyz': list(im.GetSpacing()),
            'origin_xyz': list(im.GetOrigin()), 'direction': list(im.GetDirection()),
            'present': present}
    with open(base + '_meta.json.tmp', 'w') as f:  # written last: marks the case as done
        json.dump(meta, f)
    os.replace(base + '_meta.json.tmp', base + '_meta.json')
    return base, 'done'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--num_classes', type=int, default=7)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--format', choices=['b2nd', 'npy'], default='b2nd')
    a = p.parse_args(argv)
    for mode, il, ll in (('train', 'imagesTr', 'labelsTr'), ('val', 'imagesVal', 'labelsVal')):
        images = sorted(glob(os.path.join(a.data_dir, il, '*.nii.gz')))
        if a.limit:
            images = images[:a.limit]
        out = os.path.join(a.cache_dir, mode)
        os.makedirs(out, exist_ok=True)
        jobs = []
        for im in images:
            lb = label_path_for(im, il, ll)
            if lb is None:
                raise FileNotFoundError("no label for %s" % im)
            jobs.append((im, lb, out, a.num_classes, a.format))
        print("%s: %d cases" % (mode, len(jobs)), flush=True)
        with Pool(a.workers) as pool:
            for i, (base, st) in enumerate(pool.imap_unordered(convert, jobs)):
                if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                    print("  %d/%d (%s)" % (i + 1, len(jobs), st), flush=True)


if __name__ == '__main__':
    main()
