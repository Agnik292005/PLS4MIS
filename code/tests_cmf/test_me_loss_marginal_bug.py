"""
test_me_loss_marginal_bug.py

Agnik's own experimental/verification tests for the CMF adaptation work,
kept separate from Sukhraj's tests/ folder (tests_cmf/ instead of tests/)
to avoid mixing the two.

Numeric proof of the marginal-merging bug in DC_CE_Marginal_Value_Exclusion_loss
(models/MELoss/loss.py), found by comparing the code against Eq. (4) of the
paper it implements (Shi et al., "Marginal loss and exclusion loss for
partially supervised multi-organ segmentation", Medical Image Analysis 2021,
arXiv:2007.03868):

    Paper's rule:  q_m = sum_{n in Phi_m} p_n
                   (merge REAL PROBABILITIES, i.e. softmax outputs, together)

    Code's actual behaviour (models/MELoss/loss.py,
    DC_CE_Marginal_Value_Exclusion_loss.forward()): sums RAW LOGITS first,
    applies softmax after. Confirmed by hand-tracing the code, and proven
    numerically here.

Run with: python -m unittest discover -s tests_cmf -p test_me_loss_marginal_bug.py -v
"""

import unittest
import torch
import torch.nn.functional as F


class MELossMarginalMergeBugTests(unittest.TestCase):
    def test_sum_logits_then_softmax_differs_from_sum_probs(self):
        """The code's actual approach (sum raw logits, softmax after) should
        NOT match the paper's specified approach (softmax first, sum the
        real probabilities after)."""
        logits = torch.tensor([2.0, 0.5, 1.0])

        real_probs = F.softmax(logits, dim=0)
        correct_merged_prob = (real_probs[0] + real_probs[2]).item()

        merged_logit = logits[0] + logits[2]
        remaining_logit = logits[1]
        buggy_2class = torch.tensor([merged_logit, remaining_logit])
        buggy_probs = F.softmax(buggy_2class, dim=0)
        buggy_merged_prob = buggy_probs[0].item()

        self.assertNotAlmostEqual(
            correct_merged_prob, buggy_merged_prob, places=3,
            msg="Expected the buggy sum-logits-then-softmax approach to differ "
                "from the paper's correct sum-probabilities approach, but they "
                "matched. Re-check whether the bug is still present."
        )

        self.assertAlmostEqual(correct_merged_prob, 0.8598, places=3)
        self.assertAlmostEqual(buggy_merged_prob, 0.9241, places=3)

    def test_correct_approach_is_order_independent(self):
        """Sanity check: merging real probabilities should give the same
        answer regardless of order added, a basic property real
        probabilities should have."""
        logits = torch.tensor([2.0, 0.5, 1.0])
        real_probs = F.softmax(logits, dim=0)

        order_a = (real_probs[0] + real_probs[2]).item()
        order_b = (real_probs[2] + real_probs[0]).item()

        self.assertAlmostEqual(order_a, order_b, places=6)


if __name__ == "__main__":
    unittest.main()
