import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.cmf_partial_losses import make_loss


class TestLossNames(unittest.TestCase):
    def test_old_names_are_aliases(self):
        pairs = [('me_stock', 'me_pl4mis'), ('me_fixed', 'me_pl4cmf'), ('mt_fixed', 'mt_pl4cmf'),
                 ('plseg_stock', 'plseg_pl4mis'), ('plseg_hybrid', 'plseg_pl4cmf'), ('leaf', 'leaf_pl4cmf')]
        for old, new in pairs:
            self.assertIs(type(make_loss(old, 7)), type(make_loss(new, 7)), old)

    def test_mt_stock_is_reserved(self):
        with self.assertRaises(ValueError):
            make_loss('mt_stock', 7)

    def test_plseg_variants_differ(self):
        self.assertFalse(make_loss('plseg_pl4mis', 7).hadf[0].scan_level)
        self.assertTrue(make_loss('plseg_pl4cmf', 7).hadf[0].scan_level)


if __name__ == '__main__':
    unittest.main()
