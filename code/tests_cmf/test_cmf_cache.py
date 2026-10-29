import os, tempfile, unittest
import numpy as np
import SimpleITK as sitk
import torch
from data_prep import make_cmf_cache as M
from dataloader.cmf_cache_dataset import CMFCacheDataset, EpochShuffleSampler, random_crop_padded
from dataloader.cmf_case_dataset import CMFCaseDataset
from dataloader.cmf_transforms import ZScore, PadToPatch
from dataloader.cmf_task_select import pick_task
from tests_cmf.test_train_cmf_smoke import _make
import train_DoDNet_cmf as T


class Cache(unittest.TestCase):
    FMT = 'npy'

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = os.path.join(cls.tmp.name, 'data')
        cls.cache = os.path.join(cls.tmp.name, 'cache')
        _make(cls.data)
        M.main(['--data_dir', cls.data, '--cache_dir', cls.cache, '--workers', '2', '--format', cls.FMT])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cache_equals_old_pipeline(self):
        ds = CMFCacheDataset(self.cache, 'train', (40, 40, 32), 7, 42)
        old = CMFCaseDataset(self.data, 'train', None, 7, 42)
        for i in range(len(ds)):
            s = ds[(i, 0)]
            o = PadToPatch([40, 40, 32])(ZScore()(old[i]))
            np.testing.assert_allclose(s['image'][0].numpy(), o['image'], atol=1e-5)
            np.testing.assert_array_equal(s['label'].numpy(), o['label'])

    def test_task_matches_selector_and_is_present(self):
        ds = CMFCacheDataset(self.cache, 'train', (32, 32, 32), 7, 42)
        for i, row in enumerate(ds.rows):
            for ep in range(6):
                s = ds[(i, ep)]
                self.assertEqual(int(s['forced_class']), pick_task(row['present'], row['name'], ep, 42))
                self.assertIn(int(s['forced_class']), row['present'])

    def test_val_returns_full_volume(self):
        ds = CMFCacheDataset(self.cache, 'val', (32, 32, 32), 7, 42)
        s = ds[0]
        self.assertEqual(tuple(s['image'].shape), (1,) + tuple(s['label'].shape))
        self.assertAlmostEqual(float(s['image'].mean()), 0.0, places=4)

    def test_sampler_depends_on_epoch_and_is_reproducible(self):
        a, b = EpochShuffleSampler(20, 42), EpochShuffleSampler(20, 42)
        self.assertEqual(list(a), list(b))
        b.set_epoch(1)
        self.assertNotEqual([i for i, _ in a], [i for i, _ in b])
        self.assertTrue(all(e == 1 for _, e in b))

    def test_crop_shapes(self):
        img = np.random.randn(20, 40, 40).astype(np.float32)
        lab = np.ones((20, 40, 40), np.uint8)
        ci, cl = random_crop_padded(img, lab, (32, 32, 32))
        self.assertEqual(ci.shape, (32, 32, 32))
        self.assertEqual(int((cl == 0).sum()), 32 * 32 * 32 - 20 * 32 * 32)

    def test_train_from_cache_and_resume(self):
        snap = os.path.join(self.tmp.name, 'snap')
        base = ['--cache_dir', self.cache, '--snapshot_dir', snap, '--patch_size', '32', '32', '32',
                '--device', 'cpu', '--workers', '2', '--val_interval', '1', '--save_interval', '1',
                '--final_val_overlap', '0.0']
        T.main(base + ['--epoches', '2'])
        T.main(base + ['--epoches', '3', '--resume', os.path.join(snap, 'latest.pth')])
        with open(os.path.join(snap, 'task_log.csv')) as f:
            lines = f.read().strip().split('\n')[1:]
        self.assertEqual(sorted({int(l.split(',')[0]) for l in lines}), [0, 1, 2])
        self.assertTrue(all(int(l.split(',')[2]) >= 1 for l in lines))


class CacheB2nd(Cache):
    FMT = 'b2nd'

    def test_files_are_b2nd(self):
        names = os.listdir(os.path.join(self.cache, 'train'))
        self.assertTrue(any(n.endswith('_img.b2nd') for n in names))
        self.assertFalse(any(n.endswith('.npy') for n in names))


if __name__ == '__main__':
    unittest.main()
