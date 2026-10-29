import os
import random
import shutil
import tempfile
import unittest
import numpy as np
import SimpleITK as sitk
from dataloader import transforms as tr
from dataloader import cmf_transforms as ct
from dataloader.cmf_case_dataset import CMFCaseDataset, label_path_for

NC = 7


def write_nii(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(arr), path)


def make_case(root, split, name, classes, label_with_suffix, shape=(12, 20, 24), seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(300, 80, size=shape).astype(np.float32)
    lab = np.zeros(shape, dtype=np.uint8)
    for i, c in enumerate(classes):
        lab[:, :, i * 3:(i + 1) * 3] = c
    s = 'Tr' if split == 'train' else 'Val'
    write_nii(os.path.join(root, 'images' + s, name + '_0000.nii.gz'), img)
    lname = name + ('_0000' if label_with_suffix else '') + '.nii.gz'
    write_nii(os.path.join(root, 'labels' + s, lname), lab)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        make_case(self.root, 'train', 'c1', (1, 2, 3, 4, 5), label_with_suffix=False, seed=1)
        make_case(self.root, 'train', 'c2', (1, 4), label_with_suffix=True, seed=2)   # suffixed label
        make_case(self.root, 'train', 'c3', (6,), label_with_suffix=False, seed=3)    # lesion only
        make_case(self.root, 'val', 'v1', (1, 2), label_with_suffix=False, seed=4)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_one_row_per_case(self):
        ds = CMFCaseDataset(self.root, 'train', num_classes=NC)
        self.assertEqual(len(ds), 3)

    def test_label_lookup_both_naming_styles(self):
        img = os.path.join(self.root, 'imagesTr', 'c2_0000.nii.gz')
        self.assertTrue(label_path_for(img, 'imagesTr', 'labelsTr').endswith('c2_0000.nii.gz'))
        img = os.path.join(self.root, 'imagesTr', 'c1_0000.nii.gz')
        self.assertTrue(label_path_for(img, 'imagesTr', 'labelsTr').endswith('c1.nii.gz'))

    def test_missing_label_raises(self):
        os.remove(os.path.join(self.root, 'labelsTr', 'c1.nii.gz'))
        with self.assertRaises(FileNotFoundError):
            CMFCaseDataset(self.root, 'train', num_classes=NC)

    def test_forced_class_comes_from_present_classes_and_cycles(self):
        ds = CMFCaseDataset(self.root, 'train', num_classes=NC)
        idx = {os.path.basename(r['image']): i for i, r in enumerate(ds.rows)}
        for name, present in (('c1_0000.nii.gz', [1, 2, 3, 4, 5]), ('c2_0000.nii.gz', [1, 4]), ('c3_0000.nii.gz', [6])):
            picks = []
            for e in range(10):
                ds.set_epoch(e)
                picks.append(ds[idx[name]]['forced_class'])
            self.assertTrue(set(picks) <= set(present))
            k = len(present)
            self.assertEqual(sorted(picks[:k]), present)          # first block hits each once
        # lesion-only case always picks the lesion
        ds.set_epoch(7)
        self.assertEqual(ds[idx['c3_0000.nii.gz']]['forced_class'], 6)

    def test_val_mode_has_no_forced_class(self):
        ds = CMFCaseDataset(self.root, 'val', num_classes=NC)
        self.assertEqual(len(ds), 1)
        self.assertNotIn('forced_class', ds[0])

    def test_full_pipeline_with_short_axis(self):
        patch = (16, 16, 16)   # (x, y, z); cases are z=12 so z must be padded
        tfs = [ct.ZScore(), ct.PadToPatch(patch), tr.LabeledClass(NC), tr.TrainerCrop(patch),
               tr.CreateOnehotLabel(NC), ct.ToTensorZ()]

        def compose(s):
            for t in tfs:
                s = t(s)
            return s
        random.seed(0)
        ds = CMFCaseDataset(self.root, 'train', transform=compose, num_classes=NC)
        for i in range(len(ds)):
            s = ds[i]
            self.assertEqual(tuple(s['image'].shape), (1, 16, 16, 16))
            self.assertEqual(tuple(s['onehot_label'].shape), (NC, 16, 16, 16))
            self.assertIn(int(s['forced_class']), range(1, NC))
            # the chosen class is one that the full-volume label really contains
            present = np.flatnonzero(s['cur_task'].numpy()) + 1
            self.assertIn(int(s['forced_class']), present.tolist())


if __name__ == '__main__':
    unittest.main()
