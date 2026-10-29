import unittest
import numpy as np
from utils import cmf_eval as ev


def vol(shape=(10, 10, 10)):
    return np.zeros(shape, dtype=np.int16)


class TestPerClassDice(unittest.TestCase):
    def test_perfect_and_disjoint_and_half(self):
        gt = vol(); gt[:, :, :4] = 1; gt[:, :, 6:] = 2
        pred = vol(); pred[:, :, :4] = 1            # class 1 perfect, class 2 missed
        d = ev.per_class_dice(pred, gt, classes=(1, 2))
        self.assertAlmostEqual(d[0], 1.0)
        self.assertAlmostEqual(d[1], 0.0)
        half = vol(); half[:5, :, :4] = 1            # half of class 1
        d = ev.per_class_dice(half, gt, classes=(1,))
        self.assertAlmostEqual(d[0], 2 * 200 / (200 + 400))

    def test_class_absent_from_gt_is_nan_even_if_predicted(self):
        gt = vol(); gt[:, :, :4] = 1
        pred = vol(); pred[:, :, :4] = 1; pred[:, :, 8:] = 6   # false lesion
        d = ev.per_class_dice(pred, gt)
        self.assertTrue(np.isnan(d[ev.FOREGROUND.index(6)]))   # not scored
        self.assertTrue(np.isnan(d[ev.FOREGROUND.index(2)]))
        self.assertAlmostEqual(d[0], 1.0)

    def test_background_never_scored(self):
        self.assertNotIn(0, ev.FOREGROUND)
        self.assertEqual(len(ev.per_class_dice(vol(), vol())), 6)

    def test_empty_vs_empty_is_nan_not_one(self):
        d = ev.per_class_dice(vol(), vol())
        self.assertTrue(np.isnan(d).all())


class TestStockBehaviourDocumented(unittest.TestCase):
    def test_stock_scores_empty_vs_empty_as_perfect(self):
        """Why the stock validation is wrong for partial labels."""
        try:
            from utils.evaluation_seg import get_multi_class_evaluation_score
        except Exception as e:  # medpy/scipy missing
            self.skipTest("stock evaluation not importable: %r" % (e,))
        gt = vol(); gt[:, :, :4] = 1
        pred = vol(); pred[:, :, :4] = 1
        stock = get_multi_class_evaluation_score(pred, gt, [0, 1, 2], False, 'dice')
        self.assertAlmostEqual(stock[2], 1.0, places=3)    # absent class counted as perfect
        self.assertGreater(stock[0], 0.9)                  # background inflates the mean


class TestMeter(unittest.TestCase):
    def test_nan_skipped_in_average(self):
        m = ev.ClassMeter(classes=(1, 2))
        m.update([1.0, np.nan]); m.update([np.nan, np.nan]); m.update([0.5, 0.2])
        self.assertAlmostEqual(m.avg[0], 0.75)
        self.assertEqual(m.count.tolist(), [2, 1])
        self.assertAlmostEqual(m.avg[1], 0.2)

    def test_class_never_seen_is_nan_and_excluded_from_means(self):
        m = ev.ClassMeter()
        m.update(ev.per_class_dice(*self._anatomy_case()))
        s = m.summary()
        self.assertTrue(np.isnan(s['lesion']))
        self.assertEqual(s['n_cases'][6], 0)
        self.assertAlmostEqual(s['mean_all'], s['anatomy_mean'])

    @staticmethod
    def _anatomy_case():
        gt = vol((6, 6, 30))
        for c in (1, 2, 3, 4, 5):
            gt[:, :, (c - 1) * 6:c * 6] = c
        return gt.copy(), gt

    @staticmethod
    def _lesion_case(correct):
        gt = vol(); gt[2:4, 2:4, 2:4] = 6
        pred = gt.copy() if correct else vol()
        return pred, gt

    def test_mixed_cases_and_summary(self):
        m = ev.ClassMeter()
        m.update(ev.per_class_dice(*self._anatomy_case()))          # anatomy 1..5 all 1.0
        m.update(ev.per_class_dice(*self._lesion_case(True)))        # lesion 1.0
        m.update(ev.per_class_dice(*self._lesion_case(False)))       # lesion 0.0
        s = m.summary()
        self.assertEqual(s['n_cases'], {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2})
        self.assertAlmostEqual(s['anatomy_mean'], 1.0)
        self.assertAlmostEqual(s['lesion'], 0.5)
        self.assertAlmostEqual(s['mean_all'], (5 * 1.0 + 0.5) / 6)   # six classes, not 7

    def test_lesion_only_case_does_not_touch_anatomy(self):
        m = ev.ClassMeter()
        pred, gt = self._lesion_case(True)
        pred = pred.copy(); pred[8:, 8:, 8:] = 2                     # stray anatomy prediction
        m.update(ev.per_class_dice(pred, gt))
        self.assertEqual(m.count.tolist(), [0, 0, 0, 0, 0, 1])


if __name__ == '__main__':
    unittest.main()
