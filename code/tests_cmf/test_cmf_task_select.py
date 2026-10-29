import collections
import unittest
from dataloader.cmf_task_select import pick_task


class TestPickTask(unittest.TestCase):
    def test_each_block_contains_every_task_once(self):
        for k in range(1, 7):
            tasks = list(range(1, k + 1))
            for case in ("a", "case_17", "ToothFairy3P_004"):
                for block in range(20):
                    picks = [pick_task(tasks, case, block * k + j) for j in range(k)]
                    self.assertEqual(sorted(picks), tasks, (k, case, block))

    def test_exact_counts_over_250_epochs(self):
        c = collections.Counter(pick_task([1, 2, 3, 4, 5], "x", e) for e in range(250))
        self.assertEqual(set(c.values()), {50})
        c = collections.Counter(pick_task([1, 2, 4, 5], "x", e) for e in range(250))
        self.assertTrue(set(c.values()) <= {62, 63})
        self.assertEqual(sum(c.values()), 250)
        c = collections.Counter(pick_task([6], "lesion_case", e) for e in range(250))
        self.assertEqual(c, {6: 250})

    def test_pure_function_resume_gives_same_picks(self):
        full = [pick_task([1, 2, 3, 4, 5], "x", e, 42) for e in range(250)]
        again = {e: pick_task([5, 4, 3, 2, 1], "x", e, 42) for e in reversed(range(120, 250))}
        self.assertEqual([again[e] for e in range(120, 250)], full[120:])

    def test_seed_and_case_change_the_order(self):
        a = [pick_task([1, 2, 3, 4, 5], "x", e, 42) for e in range(50)]
        b = [pick_task([1, 2, 3, 4, 5], "x", e, 123) for e in range(50)]
        c = [pick_task([1, 2, 3, 4, 5], "y", e, 42) for e in range(50)]
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_not_a_fixed_rotation(self):
        # across blocks the order must differ (otherwise it is not shuffled)
        orders = {tuple(pick_task([1, 2, 3, 4, 5], "x", b * 5 + j) for j in range(5)) for b in range(30)}
        self.assertGreater(len(orders), 10)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            pick_task([], "x", 0)


if __name__ == '__main__':
    unittest.main()
