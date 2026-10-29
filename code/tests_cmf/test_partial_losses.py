"""CPU tests for models/cmf_partial_losses.py."""
import unittest
import torch
from models.cmf_partial_losses import make_loss, MarginalExclusionCMF, LeafDiceCMF

C = 7


def _case(seed, present, bg=True, shape=(6, 7, 8), B=1):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((B, C) + shape, generator=g).requires_grad_(True)
    lab = torch.zeros((B,) + shape, dtype=torch.long)
    for k, c in enumerate(present):                     # one block per class
        lab[:, k % shape[0], :, k:k + 2] = c
    if not bg:
        lab[lab == 0] = present[0]
    ann = torch.zeros(B, C)
    ann[:, list(present)] = 1.0
    return logits, lab, ann


class Losses(unittest.TestCase):
    def test_me_modes_agree_when_nothing_is_unannotated(self):
        # every class annotated and present in the crop: logit-merge and probability-merge have nothing to merge
        logits, lab, ann = _case(0, [1, 2, 3, 4, 5, 6])
        a = make_loss('me_stock', C)(logits, lab, ann)
        b = make_loss('me_fixed', C)(logits, lab, ann)
        self.assertAlmostEqual(float(a), float(b), places=5)
        ga, = torch.autograd.grad(a, logits, retain_graph=True)
        gb, = torch.autograd.grad(b, logits)
        self.assertTrue(torch.allclose(ga, gb, atol=1e-6))

    def test_me_modes_differ_when_something_is_unannotated(self):
        logits, lab, ann = _case(1, [1, 4])
        a = make_loss('me_stock', C)(logits, lab, ann)
        b = make_loss('me_fixed', C)(logits, lab, ann)
        self.assertGreater(abs(float(a) - float(b)), 1e-4)

    def test_losses_are_finite_and_backprop(self):
        for name in ('me_stock', 'me_fixed', 'leaf'):
            logits, lab, ann = _case(2, [2, 3, 5], B=2)
            loss = make_loss(name, C)(logits, lab, ann)
            self.assertTrue(torch.isfinite(loss), name)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all(), name)
            self.assertGreater(float(logits.grad.abs().sum()), 0, name)

    def test_stock_loss_with_empty_crop_still_backprops(self):
        logits = torch.randn((1, C, 4, 4, 4), requires_grad=True)
        lab = torch.zeros((1, 4, 4, 4), dtype=torch.long)
        loss = make_loss('me_stock', C)(logits, lab, torch.zeros(1, C))
        loss.backward()                                   # must not raise
        self.assertEqual(float(loss), 0.0)

    def test_leaf_equals_repo_leafdice_when_all_annotated(self):
        from models.LeafDice.leaf_dice_loss import LeafDiceLoss
        logits, lab, ann = _case(3, [1, 2, 3, 4, 5, 6])
        mine = LeafDiceCMF(C)(logits, lab, ann)
        repo = LeafDiceLoss(C)(logits, lab)
        self.assertAlmostEqual(float(mine), float(repo), places=5)


class MultiTalentTests(unittest.TestCase):
    def test_scan_level_equals_repo_when_annotated_is_what_the_crop_shows(self):
        from models.MultiTalent.multitalent import SampleClassAdaptiveLoss
        for B in (1, 2):
            logits, lab, ann = _case(4, [1, 3, 5], B=B)
            repo = SampleClassAdaptiveLoss()(logits, lab)
            mine = make_loss('mt_fixed', C)(logits, lab, ann)
            wrap = make_loss('mt_stock', C)(logits, lab, ann)
            self.assertAlmostEqual(float(repo), float(mine), places=4)
            self.assertAlmostEqual(float(repo), float(wrap), places=6)
            g1, = torch.autograd.grad(repo, logits, retain_graph=True)
            g2, = torch.autograd.grad(mine, logits)
            self.assertTrue(torch.allclose(g1, g2, atol=1e-6))

    def test_class_annotated_in_scan_but_missing_from_crop_changes_the_loss(self):
        logits, lab, ann = _case(5, [1, 3])
        ann2 = ann.clone()
        ann2[:, 5] = 1.0                                   # class 5 annotated in the scan, not in this crop
        a = make_loss('mt_fixed', C)(logits, lab, ann)
        b = make_loss('mt_fixed', C)(logits, lab, ann2)
        self.assertGreater(abs(float(a) - float(b)), 1e-4)


def _authors_hybrid():
    """The authors' HADFLossHybrid cannot run as shipped (it adds in place to a leaf tensor that requires grad).
    Their source with exactly that one construct fixed, used as the reference."""
    import inspect, types
    import models.Proposed.sing_class_loss as m
    src = inspect.getsource(m)
    assert src.count("requires_grad=True, device=device)") == 2
    src = src.replace("requires_grad=True, device=device)", "requires_grad=True, device=device) * 1.0")
    mod = types.ModuleType('hadf_reference')
    exec(compile(src, 'hadf_reference', 'exec'), mod.__dict__)
    return mod.HADFLossHybrid


class PLSegTests(unittest.TestCase):
    @staticmethod
    def _probs(logits):
        return torch.clamp(torch.softmax(logits, dim=1), min=1e-10, max=1.0)

    @staticmethod
    def _onehot(lab):
        return torch.nn.functional.one_hot(lab, C).permute(0, 4, 1, 2, 3).float()

    def test_ports_equal_the_authors_hybrid_loss(self):
        # The authors' HADFLossHybrid (one line fixed, see _authors_hybrid). The stock port must equal it when every visible class is
        # annotated (nothing to skip, nothing weak), and the hybrid port must equal it in general (incl. weak supervision).
        from models.cmf_partial_losses import HADFCMF
        HADFLossHybrid = _authors_hybrid()
        stock, hyb, ref_a, ref_b = HADFCMF(C), HADFCMF(C, scan_level=True), HADFLossHybrid(C), HADFLossHybrid(C)
        for step in range(4):                                   # several calls: the 50-step history must agree too
            logits, lab, ann = _case(10 + step, [1, 3, 5], B=2)
            ann_wide = ann.clone()
            ann_wide[:, 6] = 1.0                                # annotated in the scan, not visible in the crop
            p, t = self._probs(logits), self._onehot(lab)
            a = stock(p, t)
            b = ref_a(p, t, ann[:, 1:])
            self.assertAlmostEqual(float(a), float(b), places=4)
            c_ = hyb(p, t, ann_wide)
            d = ref_b(p, t, ann_wide[:, 1:])
            self.assertAlmostEqual(float(c_), float(d), places=4)
            ga, = torch.autograd.grad(c_, logits, retain_graph=True)
            gb, = torch.autograd.grad(d, logits)
            self.assertTrue(torch.allclose(ga, gb, atol=1e-5))

    def test_total_loss_runs_and_backprops(self):
        from models.cmf_partial_losses import PLSegLoss
        for scan_level in (False, True):
            crit = PLSegLoss(C, scan_level=scan_level, epochs=10)
            g = torch.Generator().manual_seed(0)
            sizes = (16, 8, 4, 2)
            outs = [torch.randn((1, C, s, s, s), generator=g).requires_grad_(True) for s in sizes]
            lab = torch.zeros((1, 16, 16, 16), dtype=torch.long)
            lab[:, :6, :, :6] = 2
            lab[:, 8:, 8:, 8:] = 5
            ann = torch.zeros(1, C)
            ann[:, [2, 5, 6]] = 1.0
            loss, parts = crit(outs, lab, ann, 3)
            self.assertTrue(torch.isfinite(loss) and all(v == v for v in parts.values()))
            loss.backward()
            for o in outs:
                self.assertTrue(torch.isfinite(o.grad).all())
                self.assertGreater(float(o.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
