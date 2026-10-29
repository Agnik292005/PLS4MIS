"""
cmf_task_select.py

Shuffled-cycle task selection for DoDNet (PLAN.md section 3, one row per case,
one eligible task per visit).

Epochs are grouped into blocks of k = number of eligible tasks of the case.
Each block is a fresh shuffle of those tasks, so every task is picked exactly
once per k epochs. The pick is a pure function of (seed, case, epoch), so a
resumed run makes the same picks as an uninterrupted one.
"""
import hashlib
import random


def _block_seed(base_seed, case_name, block):
    h = hashlib.sha256(("%s|%s|%d" % (base_seed, case_name, block)).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def pick_task(tasks, case_name, epoch, base_seed=42):
    """tasks: iterable of class ids present in this case (any order).
    Returns one of them."""
    tasks = sorted(int(t) for t in set(tasks))
    k = len(tasks)
    if k == 0:
        raise ValueError("no eligible task for case %s" % case_name)
    if k == 1:
        return tasks[0]
    block, pos = divmod(int(epoch), k)
    order = list(tasks)
    random.Random(_block_seed(base_seed, case_name, block)).shuffle(order)
    return order[pos]
