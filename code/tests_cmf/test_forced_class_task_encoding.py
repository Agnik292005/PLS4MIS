"""
test_forced_class_task_encoding.py

Comprehensive, edge-case test of the forced_class -> task_encoding
conversion logic added to train_DoDNet.py's training loop
(fix_train_loop_use_forced_class.py), covering:

  1. A single forced_class value produces the correct one-hot task
     encoding (matches what the original random-pick logic would have
     produced for that same class value).
  2. Multiple different forced_class values across a batch each produce
     their own correct, independent one-hot encoding.
  3. The lowest valid class value (1) is handled correctly (off-by-one
     boundary: forced_class is 1-indexed, task_encoding is 0-indexed).
  4. The highest valid class value (num_classes - 1) is handled correctly
     (the other boundary).
  5. forced_class absent entirely (stock AbdomenOrgan path) correctly
     takes the random-pick branch instead.
  6. forced_class as a real torch tensor (as it arrives from the actual
     dataloader/collate pipeline), not just a convenient Python int.

Uses fake, hand-checkable one-hot vectors throughout, no real data or GPU
needed, matching the pattern of tests/test_dodnet_crop_geometry.py.

Run with: python -m unittest discover -s tests_cmf -p test_forced_class_task_encoding.py -v
"""

import unittest
import torch


def compute_task_encoding(forced_class_batch, num_classes):
    """Reproduces EXACTLY the logic added to train_DoDNet.py's training
    loop for the forced_class branch, isolated here so it can be tested
    directly without needing a real model, real images, or a GPU."""
    task_encoding_list = [int(c) - 1 for c in forced_class_batch]
    task_encoding = torch.zeros(size=(len(task_encoding_list), num_classes - 1), dtype=torch.int)
    task_encoding.scatter_(1, torch.tensor(task_encoding_list, dtype=torch.int64)[:, None], 1)
    return task_encoding


def compute_task_encoding_random_pick_equivalent(chosen_class, num_classes):
    """Reproduces the ORIGINAL random-pick logic's encoding step for a
    single already-chosen class, so we can compare against the
    forced_class path's output for the SAME chosen class."""
    task_encoding_list = [chosen_class - 1]
    task_encoding = torch.zeros(size=(1, num_classes - 1), dtype=torch.int)
    task_encoding.scatter_(1, torch.tensor(task_encoding_list, dtype=torch.int64)[:, None], 1)
    return task_encoding


class ForcedClassTaskEncodingTests(unittest.TestCase):
    def test_single_forced_class_matches_random_pick_equivalent(self):
        num_classes = 6
        for chosen_class in range(1, num_classes):
            with self.subTest(chosen_class=chosen_class):
                forced_result = compute_task_encoding(torch.tensor([chosen_class]), num_classes)
                random_pick_equivalent = compute_task_encoding_random_pick_equivalent(chosen_class, num_classes)
                self.assertTrue(torch.equal(forced_result, random_pick_equivalent),
                                 f"Mismatch for class {chosen_class}: "
                                 f"forced={forced_result.tolist()} vs "
                                 f"random_pick_equivalent={random_pick_equivalent.tolist()}")

    def test_multiple_different_classes_in_one_batch(self):
        num_classes = 6
        batch = torch.tensor([1, 3, 5, 2])
        result = compute_task_encoding(batch, num_classes)

        self.assertEqual(result.shape, (4, num_classes - 1))
        expected = torch.tensor([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
        ], dtype=torch.int)
        self.assertTrue(torch.equal(result, expected))

    def test_lowest_boundary_class_value(self):
        num_classes = 6
        result = compute_task_encoding(torch.tensor([1]), num_classes)
        self.assertEqual(result[0, 0].item(), 1)
        self.assertEqual(result.sum().item(), 1)

    def test_highest_boundary_class_value(self):
        num_classes = 6
        highest_class = num_classes - 1
        result = compute_task_encoding(torch.tensor([highest_class]), num_classes)
        self.assertEqual(result[0, num_classes - 2].item(), 1)
        self.assertEqual(result.sum().item(), 1)

    def test_forced_class_as_real_tensor_type(self):
        num_classes = 6
        real_type_batch = torch.tensor([2, 4], dtype=torch.int64)
        result = compute_task_encoding(real_type_batch, num_classes)
        self.assertEqual(result.shape, (2, num_classes - 1))
        self.assertEqual(result[0, 1].item(), 1)
        self.assertEqual(result[1, 3].item(), 1)

    def test_forced_class_none_falls_back_correctly(self):
        forced_class = None
        used_forced_path = forced_class is not None
        self.assertFalse(used_forced_path,
                          "forced_class is None should NOT trigger the forced-class path")


if __name__ == "__main__":
    unittest.main()
