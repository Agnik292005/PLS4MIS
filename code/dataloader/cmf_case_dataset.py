"""
cmf_case_dataset.py

One dataset row per case (PLAN.md section 3). The task for each visit is picked
by cmf_task_select.pick_task from the classes present in the FULL label, and is
returned as `forced_class` (1-indexed class id), which the existing
transforms and training loop already carry through.

Additive: CMFDataset.py (one row per structure) and the stock AbdomenOrgan are
not touched.

Layout (same as the stock datasets):
    data_dir/imagesTr/<case>_0000.nii.gz   data_dir/labelsTr/<case>.nii.gz
    data_dir/imagesVal/...                 data_dir/labelsVal/...
The label lookup tries the nnU-Net name first (channel suffix _0000 removed) and
falls back to the identical file name, because some of our folders still carry
the _0000 suffix on labels.
"""
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

from dataloader.cmf_task_select import pick_task


def label_path_for(image_path, image_dir_name, label_dir_name):
    """Return the existing label path for an image path, or None."""
    lab_dir = os.path.dirname(image_path).replace(image_dir_name, label_dir_name)
    fname = os.path.basename(image_path)
    candidates = []
    if fname.endswith("_0000.nii.gz"):
        candidates.append(os.path.join(lab_dir, fname[:-len("_0000.nii.gz")] + ".nii.gz"))
    candidates.append(os.path.join(lab_dir, fname))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class CMFCaseDataset(Dataset):
    def __init__(self, nii_dir, mode='train', transform=None, num_classes=7, seed=42):
        if mode not in ('train', 'val'):
            raise ValueError("Unsupported mode: %s" % mode)
        self.mode = mode
        self.transform = transform
        self.num_classes = num_classes
        self.seed = seed
        self.epoch = 0
        img_dir = 'imagesTr' if mode == 'train' else 'imagesVal'
        lab_dir = 'labelsTr' if mode == 'train' else 'labelsVal'
        images = sorted(glob(os.path.join(nii_dir, img_dir, '*.nii.gz')))
        if not images:
            raise FileNotFoundError("no images in %s" % os.path.join(nii_dir, img_dir))
        self.rows, missing = [], []
        for im in images:
            lb = label_path_for(im, img_dir, lab_dir)
            if lb is None:
                missing.append(os.path.basename(im))
            else:
                self.rows.append({'image': im, 'label': lb})
        if missing:
            raise FileNotFoundError("%d images have no label, e.g. %s" % (len(missing), missing[:5]))
        print("CMFCaseDataset[%s]: %d cases (one row per case)" % (mode, len(self.rows)))

    def set_epoch(self, epoch):
        """Call once per epoch BEFORE iterating the DataLoader (workers are
        re-created each epoch, so they pick up the new value)."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = sitk.GetArrayFromImage(sitk.ReadImage(row['image']))
        label = sitk.GetArrayFromImage(sitk.ReadImage(row['label']))
        name = os.path.basename(row['image'])
        sample = {'image': image, 'label': label, 'img_name': name}
        if self.mode == 'train':
            present = [int(c) for c in np.unique(label) if 0 < c < self.num_classes]
            if not present:
                raise ValueError("case %s has no foreground label" % name)
            sample['forced_class'] = pick_task(present, name, self.epoch, self.seed)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
