"""
cmf_eval.py

Annotation-aware Dice for the CMF benchmark (PLAN.md section 4).
Additive: the stock utils/evaluation_seg.py is not touched.

Rules:
  * Background (class 0) is never scored.
  * A class is scored for a case ONLY if it is present in that case's ground
    truth. If it is absent (not annotated, or not in the field of view) the
    score is NaN and is skipped, never counted as 0 or 1.
  * Per-class averages are taken over the cases where the class was scoreable.
  * "mean_all" is the mean of the per-class averages over the classes that had
    at least one scoreable case (six classes in the full 7-class setup). It is
    the number used to pick the best checkpoint.
"""
import numpy as np

ANATOMY = (1, 2, 3, 4, 5)
LESION = 6
FOREGROUND = (1, 2, 3, 4, 5, 6)


def per_class_dice(pred, gt, classes=FOREGROUND):
    """Dice for each class in `classes`; NaN where the class is not in `gt`."""
    out = np.full(len(classes), np.nan, dtype=np.float64)
    for i, c in enumerate(classes):
        g = gt == c
        g_sum = int(g.sum())
        if g_sum == 0:
            continue
        p = pred == c
        out[i] = 2.0 * int(np.logical_and(p, g).sum()) / (int(p.sum()) + g_sum)
    return out


class ClassMeter(object):
    """Running per-class average that skips NaN (unscoreable) entries."""

    def __init__(self, classes=FOREGROUND):
        self.classes = tuple(classes)
        self.sum = np.zeros(len(self.classes), dtype=np.float64)
        self.count = np.zeros(len(self.classes), dtype=np.int64)

    def update(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        ok = ~np.isnan(scores)
        self.sum[ok] += scores[ok]
        self.count[ok] += 1

    @property
    def avg(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self.count > 0, self.sum / np.maximum(self.count, 1), np.nan)

    def summary(self):
        avg = self.avg
        per_class = {c: float(a) for c, a in zip(self.classes, avg)}
        n = {c: int(k) for c, k in zip(self.classes, self.count)}

        def _mean(cs):
            vals = [per_class[c] for c in cs if c in per_class and not np.isnan(per_class[c])]
            return float(np.mean(vals)) if vals else float('nan')

        return {
            'per_class': per_class,
            'n_cases': n,
            'anatomy_mean': _mean(ANATOMY),
            'lesion': per_class.get(LESION, float('nan')),
            'mean_all': _mean(self.classes),
        }
