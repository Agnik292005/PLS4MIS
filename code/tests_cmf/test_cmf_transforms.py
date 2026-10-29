import random
import unittest
import numpy as np
from dataloader import transforms as tr
from dataloader import cmf_transforms as ct

NC = 6  # classes incl. background


def make_sample(shape, classes=(1, 4), seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(500, 200, size=shape).astype(np.float32)
    lab = np.zeros(shape, dtype=np.int16)
    z, y, x = shape
    for i, c in enumerate(classes):
        lab[:, :, i * (x // 4):(i + 1) * (x // 4)] = c
    return {'image': img, 'label': lab, 'img_name': 'case.nii.gz'}


def train_pipeline(patch):
    return [ct.ZScore(), ct.PadToPatch(patch), tr.LabeledClass(NC),
            tr.TrainerCrop(patch), tr.CreateOnehotLabel(NC), ct.ToTensorZ()]


def run(tfs, s):
    for t in tfs:
        s = t(s)
    return s


class TestZScore(unittest.TestCase):
    def test_whole_volume_mean0_std1(self):
        s = ct.ZScore()(make_sample((30, 40, 40)))
        self.assertAlmostEqual(float(s['image'].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(s['image'].std()), 1.0, places=4)

    def test_constant_image_raises(self):
        s = make_sample((8, 8, 8)); s['image'][:] = 3
        with self.assertRaises(ValueError):
            ct.ZScore()(s)

    def test_input_not_modified(self):
        s = make_sample((8, 8, 8)); before = s['image'].copy()
        ct.ZScore()(s)
        np.testing.assert_array_equal(s['image'], before)


class TestPad(unittest.TestCase):
    def test_short_z_axis_padded_with_zero(self):
        # array (z=170,y,x); patch (x,y,z)=(32,32,192) -> needs z=192
        s = ct.ZScore()(make_sample((170, 40, 40)))
        out = ct.PadToPatch((32, 32, 192))(s)
        self.assertEqual(out['image'].shape, (192, 40, 40))
        self.assertEqual(out['label'].shape, (192, 40, 40))
        self.assertTrue((out['image'][:11] == 0).all())      # 22 missing -> 11 each side
        self.assertTrue((out['image'][-11:] == 0).all())
        self.assertTrue((out['label'][:11] == 0).all())
        np.testing.assert_array_equal(out['image'][11:181], s['image'])  # real data untouched

    def test_noop_when_big_enough(self):
        s = make_sample((50, 50, 50))
        out = ct.PadToPatch((32, 32, 32))(s)
        self.assertIs(out['image'], s['image'])

    def test_odd_deficit(self):
        s = make_sample((21, 40, 40))
        out = ct.PadToPatch((32, 32, 32))(s)
        self.assertEqual(out['image'].shape[0], 32)

    def test_label_content_preserved(self):
        s = make_sample((10, 40, 40), classes=(1, 2, 3))
        out = ct.PadToPatch((32, 32, 32))(s)
        self.assertEqual(sorted(np.unique(out['label'])), [0, 1, 2, 3])
        for c in (1, 2, 3):
            self.assertEqual((out['label'] == c).sum(), (s['label'] == c).sum())


class TestStockBug(unittest.TestCase):
    def test_stock_trainercrop_crashes_on_short_axis(self):
        """Documents why padding is needed: stock TrainerCrop has no padding."""
        s = tr.LabeledClass(NC)(make_sample((10, 40, 40)))
        with self.assertRaises(ValueError):
            tr.TrainerCrop((32, 32, 16))(s)


class TestPipeline(unittest.TestCase):
    def test_end_to_end_shapes_and_axis_order(self):
        random.seed(1)
        patch = (32, 24, 16)  # (x, y, z)
        s = run(train_pipeline(patch), make_sample((10, 30, 40)))  # z=10 short
        self.assertEqual(tuple(s['image'].shape), (1, 16, 24, 32))
        self.assertEqual(tuple(s['onehot_label'].shape), (NC, 16, 24, 32))

    def test_no_per_crop_rescale(self):
        random.seed(2)
        s = run(train_pipeline((16, 16, 16)), make_sample((30, 40, 40)))
        im = s['image'].numpy()
        # old ToTensor forced every crop into exactly [0, 1]; z-score data must not be
        self.assertLess(im.min(), 0.0)
        self.assertGreater(im.max(), 1.0)

    def test_crop_matches_whole_volume_values(self):
        random.seed(3)
        base = make_sample((30, 40, 40))
        z = ct.ZScore()(base)['image']
        s = run(train_pipeline((16, 16, 16)), base)
        im = s['image'][0].numpy()
        found = False
        for a in range(30 - 16 + 1):
            for b in range(40 - 16 + 1):
                for c in range(40 - 16 + 1):
                    if np.array_equal(z[a:a+16, b:b+16, c:c+16], im):
                        found = True
        self.assertTrue(found, "crop is not an exact piece of the z-scored volume")

    def test_task_vector_from_full_volume(self):
        random.seed(4)
        s = run(train_pipeline((8, 8, 8)), make_sample((12, 40, 40), classes=(1, 4)))
        self.assertEqual(s['cur_task'].numpy().tolist(), [1, 0, 0, 1, 0])

    def test_forced_class_passthrough(self):
        base = make_sample((12, 40, 40)); base['forced_class'] = 4
        s = run(train_pipeline((8, 8, 8)), base)
        self.assertEqual(int(s['forced_class']), 4)

    def test_val_path_zscore_only(self):
        base = make_sample((12, 40, 40))
        s = ct.Test_ToTensorZ()(ct.ZScore()(base))
        self.assertEqual(tuple(s['image'].shape), (1, 12, 40, 40))
        self.assertAlmostEqual(float(s['image'].mean()), 0.0, places=4)


if __name__ == '__main__':
    unittest.main()
