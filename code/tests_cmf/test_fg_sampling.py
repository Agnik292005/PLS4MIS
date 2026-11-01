import sys, os, tempfile, random, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataloader.cmf_cache_dataset import random_crop_padded, CMFCacheDataset
from tests_cmf.test_unet_cmf_smoke import _make_cache


class FgSampling(unittest.TestCase):
    def test_crop_contains_the_centre_voxel(self):
        img = np.zeros((40, 40, 40), np.float32)
        lab = np.zeros((40, 40, 40), np.uint8)
        for centre in [(35, 5, 20), (0, 0, 0), (39, 39, 39), (20, 20, 20)]:
            lab[:] = 0
            lab[centre] = 3
            for _ in range(5):
                _, cl = random_crop_padded(img, lab, (16, 16, 16), centre)
                self.assertTrue((cl == 3).any(), centre)

    def test_short_axis_still_works(self):
        img = np.zeros((10, 40, 40), np.float32)
        lab = np.zeros((10, 40, 40), np.uint8)
        lab[5, 30, 30] = 2
        _, cl = random_crop_padded(img, lab, (16, 16, 16), (5, 30, 30))
        self.assertEqual(cl.shape, (16, 16, 16))
        self.assertTrue((cl == 2).any())

    def test_dataset_always_contains_chosen_class_when_fg_prob_is_one(self):
        with tempfile.TemporaryDirectory() as d:
            _make_cache(d)
            ds = CMFCacheDataset(d, 'train', (32, 32, 32), 7, 42, fg_prob=1.0)
            for epoch in range(4):
                for i in range(len(ds)):
                    s = ds[(i, epoch)]
                    self.assertTrue(bool((s['label'] == s['forced_class']).any()), (i, epoch))

    def test_same_sample_same_decision(self):
        with tempfile.TemporaryDirectory() as d:
            _make_cache(d)
            ds = CMFCacheDataset(d, 'train', (32, 32, 32), 7, 42, fg_prob=1.0)
            random.seed(1); a = ds[(0, 3)]['label']
            random.seed(2); b = ds[(0, 3)]['label']
            self.assertTrue(bool((a == b).all()))

    def test_default_keeps_old_behaviour(self):
        with tempfile.TemporaryDirectory() as d:
            _make_cache(d)
            ds = CMFCacheDataset(d, 'train', (32, 32, 32), 7, 42)
            self.assertEqual(ds.fg_prob, 0.0)
            self.assertEqual(tuple(ds[(0, 0)]['image'].shape), (1, 32, 32, 32))


if __name__ == '__main__':
    unittest.main()
