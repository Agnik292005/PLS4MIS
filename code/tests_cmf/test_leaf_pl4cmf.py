import sys, os, unittest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.cmf_partial_losses import make_loss
from models.LeafDice.leaf_dice_loss import LeafDiceLoss
from tests_cmf.test_partial_losses import _case, C


class LeafPL4CMF(unittest.TestCase):
    def test_equals_repo_when_all_annotated(self):
        logits, lab, ann = _case(3, [1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(float(make_loss('leaf_pl4cmf', C)(logits, lab, ann)), float(LeafDiceLoss(C)(logits, lab)), places=5)

    def test_equals_repo_with_superset_code_when_partial(self):
        logits, lab, ann = _case(4, [1, 3])
        lab2 = torch.where(lab == 0, torch.full_like(lab, C), lab)
        self.assertAlmostEqual(float(make_loss('leaf_pl4cmf', C)(logits, lab, ann)), float(LeafDiceLoss(C)(logits, lab2)), places=5)

    def test_partial_scan_differs_from_unmapped_labels(self):
        logits, lab, ann = _case(4, [1, 3])
        self.assertGreater(abs(float(make_loss('leaf_pl4cmf', C)(logits, lab, ann)) - float(LeafDiceLoss(C)(logits, lab))), 1e-4)

    def test_merged_version_still_available(self):
        logits, lab, ann = _case(4, [1, 3])
        self.assertTrue(torch.isfinite(make_loss('leaf_merged', C)(logits, lab, ann)))


if __name__ == '__main__':
    unittest.main()
