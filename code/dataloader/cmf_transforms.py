"""
cmf_transforms.py

Extra transforms for the CMF benchmark (PLAN.md section 3). Additive: the
stock dataloader/transforms.py is not touched.

Array order everywhere is (z, y, x), as returned by sitk.GetArrayFromImage.
`patch_size` is the command-line order (x, y, z), same as the stock
TrainerCrop, which reverses it.
"""
import numpy as np
import torch


class ZScore(object):
    """Whole-volume z-score, computed ONCE on the full scan, before any crop.
    Use the same transform for training and validation/inference."""

    def __call__(self, sample):
        img = sample['image'].astype(np.float32)
        mean = img.mean(dtype=np.float64)
        std = img.std(dtype=np.float64)
        if not std > 1e-6:
            raise ValueError("constant image, cannot z-score: %s" % sample.get('img_name'))
        out = dict(sample)
        out['image'] = ((img - mean) / std).astype(np.float32)
        return out


class PadToPatch(object):
    """Pad image and label with 0 so every axis is at least the patch size.
    After z-score, 0 is the scan mean, i.e. a neutral value. Label 0 is
    background. Padding is split evenly on both sides of each short axis."""

    def __init__(self, patch_size):
        self.target = tuple(patch_size[::-1])  # to (z, y, x)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        pads = []
        for dim, t in zip(image.shape, self.target):
            deficit = max(t - dim, 0)
            before = deficit // 2
            pads.append((before, deficit - before))
        out = dict(sample)
        if any(p != (0, 0) for p in pads):
            out['image'] = np.pad(image, pads, mode='constant', constant_values=0)
            out['label'] = np.pad(label, pads, mode='constant', constant_values=0)
        return out


class ToTensorZ(object):
    """Like the stock ToTensor but WITHOUT the per-crop min-max rescaling."""

    def __call__(self, sample):
        image = sample['image'].astype(np.float32)
        image = image.reshape(1, *image.shape)
        result = {'image': torch.from_numpy(image),
                  'label': torch.from_numpy(sample['label'].astype(np.float32)),
                  'onehot_label': torch.from_numpy(sample['onehot_label']),
                  'cur_task': torch.from_numpy(sample['cur_task']),
                  'img_name': sample['img_name']}
        if 'forced_class' in sample:
            result['forced_class'] = torch.tensor(sample['forced_class'])
        return result


class Test_ToTensorZ(object):
    """Validation/inference version: add channel dim only (no rescaling)."""

    def __call__(self, sample):
        image = sample['image'].astype(np.float32)
        image = image.reshape(1, *image.shape)
        return {'img_name': sample['img_name'],
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(sample['label'].astype(np.float32))}
