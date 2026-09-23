"""
test_forced_class_survives_pipeline.py

Verifies, against the REAL transform pipeline (not a reimplementation),
that a forced_class value set by CMFAnatomyDataset actually survives all
4 transform steps (LabeledClass -> TrainerCrop -> CreateOnehotLabel ->
ToTensor) and arrives correctly at the end, confirming the coverage fix
(fix_transforms_preserve_forced_class.py) genuinely works, rather than
trusting the patch by reading it alone.

Also verifies the stock (no forced_class) path is completely unaffected,
so existing WORD/FLARE2023 runs using AbdomenOrgan are not broken by this
change.

Run with: python -m unittest discover -s tests_cmf -p test_forced_class_survives_pipeline.py -v
"""

import sys
import os
import unittest
import numpy as np
from torchvision import transforms as tv_transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import transforms as tr


class ForcedClassSurvivesPipelineTests(unittest.TestCase):
    def build_pipeline(self, num_classes, patch_size):
        return tv_transforms.Compose([
            tr.LabeledClass(num_classes),
            tr.TrainerCrop(patch_size),
            tr.CreateOnehotLabel(num_classes),
            tr.ToTensor()
        ])

    def test_forced_class_survives_all_four_transforms(self):
        num_classes = 6
        patch_size = (16, 16, 16)
        rng = np.random.default_rng(42)
        image = rng.normal(size=(32, 32, 32)).astype(np.float32)
        label = np.zeros((32, 32, 32), dtype=np.float32)
        label[5:15, 5:15, 5:15] = 3

        sample = {
            'image': image,
            'label': label,
            'img_name': 'test_case',
            'forced_class': 3,
        }

        pipeline = self.build_pipeline(num_classes, patch_size)
        result = pipeline(sample)

        self.assertIn('forced_class', result,
                       "forced_class was dropped somewhere in the pipeline")
        self.assertEqual(int(result['forced_class']), 3,
                          "forced_class value changed unexpectedly during the pipeline")

    def test_forced_class_absent_does_not_break_stock_path(self):
        num_classes = 6
        patch_size = (16, 16, 16)
        rng = np.random.default_rng(7)
        image = rng.normal(size=(32, 32, 32)).astype(np.float32)
        label = np.zeros((32, 32, 32), dtype=np.float32)
        label[5:15, 5:15, 5:15] = 2

        sample = {
            'image': image,
            'label': label,
            'img_name': 'test_case_stock',
        }

        pipeline = self.build_pipeline(num_classes, patch_size)
        result = pipeline(sample)

        self.assertNotIn('forced_class', result,
                          "forced_class appeared out of nowhere for a stock sample "
                          "that never had one")
        self.assertIn('cur_task', result)
        self.assertIn('onehot_label', result)


if __name__ == "__main__":
    unittest.main()
