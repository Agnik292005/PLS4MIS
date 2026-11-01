"""
cmf_partial_losses.py

Partial-label losses for the CMF benchmark (U-Net backbone, softmax output).
New file; models/MELoss/loss.py and models/LeafDice/*.py are not touched.

Background. In a partially labelled scan only some structures are annotated.
Everything else, including the structures that were simply not annotated, is
0 in that scan's label. So the label's 0 means "background OR a structure that
is not annotated here". Both losses below deal with that by MERGING.

me_stock   Port of models/MELoss/loss.py (Marginal Exclusion loss) with the same
           maths, minus the hard-wired .cuda(). The annotated set is read from
           the CROP (np.unique of the patch label) and the unannotated classes
           are merged into the first channel by ADDING THEIR LOGITS.
me_fixed   Same loss, two changes: the annotated set is the one of the whole
           SCAN (cache metadata 'present', passed in as `annotated`), and the
           merge ADDS PROBABILITIES (the probability of "background or any
           unannotated class" is the sum of their probabilities).
leaf       LeafDice idea (label-set generalisation of the Dice loss): per scan,
           channels = [superset = background + unannotated classes] + annotated
           classes; the superset probability is the sum of its members; mean
           (squared-denominator) Dice over these channels. Needs the scan-level set.

logits:     (B, C, D, H, W) raw network output
label:      (B, D, H, W) integer labels 0..C-1 (crop)
annotated:  (B, C) float 0/1, entry c = class c annotated in this scan (entry 0 ignored)
"""
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_DICE = 1e-5


def _dice_loss(pred, gt):
    """pred, gt: (K, N). 1 - mean over channels of 2*inter/(ref+seg+eps) (same as the stock dice_loss_mute)."""
    ref, inter, seg = gt.sum(1), (gt * pred).sum(1), pred.sum(1)
    return 1.0 - torch.mean(2.0 * inter / (ref + seg + EPS_DICE))


def _prepare(logits_i, lab_i, annotated_i, source, merge):
    """One sample. Returns kept, unann, target index (N,), merged probabilities (K, N),
    log of merged probabilities (K, N), full clamped probabilities (C, N)."""
    C = logits_i.shape[0]
    dev = logits_i.device
    if source == 'crop':
        kept = torch.unique(lab_i)
    else:
        kept = torch.tensor([0] + [c for c in range(1, C) if float(annotated_i[c]) > 0.5], device=dev, dtype=torch.long)
    is_kept = torch.zeros(C, dtype=torch.bool, device=dev)
    is_kept[kept] = True
    unann = (~is_kept).nonzero().flatten()
    lut = torch.zeros(C, dtype=torch.long, device=dev)
    lut[kept] = torch.arange(len(kept), device=dev)
    tgt = lut[lab_i].reshape(-1)
    z = logits_i.reshape(C, -1)
    probs_full = torch.softmax(z, 0).clamp(1e-10, 1.0)
    if merge == 'logits':
        zk = z[kept]
        if len(unann):
            zk = torch.cat([(zk[0] + z[unann].sum(0))[None], zk[1:]], 0)
        logp = F.log_softmax(zk, 0)
        newp = logp.exp()
    else:
        p = torch.softmax(z, 0)
        pk = p[kept]
        if len(unann):
            pk = torch.cat([(pk[0] + p[unann].sum(0))[None], pk[1:]], 0)
        newp = pk
        logp = torch.log(newp.clamp(min=1e-10))
    return kept, unann, tgt, newp, logp, probs_full


class MarginalExclusionCMF(nn.Module):
    def __init__(self, num_classes, source='case', merge='probs'):
        super().__init__()
        assert source in ('crop', 'case') and merge in ('logits', 'probs')
        self.C, self.source, self.merge = num_classes, source, merge

    def forward(self, logits, label, annotated=None):
        B = logits.shape[0]
        total = logits.sum() * 0.0
        for i in range(B):
            lab_i = label[i].long()
            if self.source == 'crop' and bool((torch.unique(lab_i) == 0).all()):
                continue                                   # stock: nothing annotated in this crop
            kept, unann, tgt, newp, logp, probs_full = _prepare(
                logits[i], lab_i, None if annotated is None else annotated[i], self.source, self.merge)
            K, N = newp.shape
            onehot = F.one_hot(tgt, K).T.float()                         # (K, N)
            ce = -logp.gather(0, tgt[None]).mean()
            dc = _dice_loss(newp, onehot)
            ex_t = torch.empty((self.C, N), device=logits.device)
            ex_t[kept] = 1.0 - onehot
            if len(unann):
                ex_t[unann] = (1.0 - onehot[0])[None].expand(len(unann), -1)
            ex = -_dice_loss(probs_full, ex_t)
            total = total + 0.5 * ce + 0.5 * dc + 0.1 * ex
        return total / B


class LeafDiceCMF(nn.Module):
    def __init__(self, num_classes, squared=True):
        super().__init__()
        self.C, self.squared = num_classes, squared

    def forward(self, logits, label, annotated):
        B = logits.shape[0]
        total = logits.sum() * 0.0
        for i in range(B):
            kept, unann, tgt, newp, logp, _ = _prepare(logits[i], label[i].long(), annotated[i], 'case', 'probs')
            onehot = F.one_hot(tgt, newp.shape[0]).T.float()
            num = (newp * onehot).sum(1)
            if self.squared:
                den = (newp * newp).sum(1) + (onehot * onehot).sum(1)
            else:
                den = newp.sum(1) + onehot.sum(1)
            total = total + (1.0 - torch.mean(2.0 * num / (den + EPS_DICE)))
        return total / B


class LeafDicePL4CMF(nn.Module):
    """Leaf-Dice as in the paper and in models/LeafDice/leaf_dice_loss.py (used unchanged). That loss needs a label code
    for the 'background or unannotated' set; nothing in this pipeline creates it. Here it is made per scan: if the scan has
    any unannotated structure, every label-0 voxel gets the superset code (= num_classes), so it is left out of the
    numerators and still counts in the denominators. A fully annotated scan keeps its labels, so background stays a class."""

    def __init__(self, num_classes):
        super().__init__()
        from models.LeafDice.leaf_dice_loss import LeafDiceLoss
        self.C = num_classes
        self.inner = LeafDiceLoss(num_classes)

    def forward(self, logits, label, annotated):
        lab = label.long().clone()
        for i in range(lab.shape[0]):
            if any(float(annotated[i][c]) < 0.5 for c in range(1, self.C)):
                lab[i] = torch.where(lab[i] == 0, torch.full_like(lab[i], self.C), lab[i])
        return self.inner(logits, lab)


class MultiTalentStock(nn.Module):
    """The repo's SampleClassAdaptiveLoss, unchanged (imported). The set of annotated classes is read from the CROP."""

    def __init__(self):
        super().__init__()
        from models.MultiTalent.multitalent import SampleClassAdaptiveLoss
        self.inner = SampleClassAdaptiveLoss()

    def forward(self, logits, label, annotated=None):
        return self.inner(logits, label.long())


class MultiTalentCMF(nn.Module):
    """Same loss as the repo's SampleClassAdaptiveLoss (per-class sigmoid, BCE + Dice, background ignored, sum over
    classes), but a class counts as annotated when it is annotated in the whole SCAN (`annotated`), not only when it
    appears in the crop. A class annotated in the scan but absent from the crop is then trained to be empty there."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits, label, annotated):
        B, C = logits.shape[:2]
        N = logits[0, 0].numel()
        target = F.one_hot(label.long(), C).permute(0, 4, 1, 2, 3).float().reshape(B, C, N)
        z = logits.reshape(B, C, N)
        prob = torch.sigmoid(z)
        m = annotated.to(logits.device).float().clone()
        m[:, 0] = 0.0
        mask = m.unsqueeze(-1).expand(-1, -1, N)
        valid = (mask.sum(dim=(0, 2)) > 0).float()
        if valid.sum() == 0:
            return (logits * 0).sum()
        bce = F.binary_cross_entropy_with_logits(z, target, reduction='none') * mask
        bce_c = bce.sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp_min(1.0)
        inter = (prob * target * mask).sum(dim=(0, 2))
        dice = 1.0 - (2.0 * inter + self.eps) / ((prob * mask).sum(dim=(0, 2)) + (target * mask).sum(dim=(0, 2)) + self.eps)
        return ((bce_c + dice) * valid).sum()


class HADFCMF(nn.Module):
    """Hardness-aware decoupled foreground loss of PL-Seg (models/Proposed/sing_class_loss.py), device-flexible port.
    Each annotated class is graded on its own as "class vs the rest" (cross-entropy + Dice); classes that do worse
    over the last 50 steps get more weight.
    scan_level=False  stock HADFLoss: graded classes = classes visible in the CROP.
    scan_level=True   the authors' HADFLossHybrid: graded classes = classes annotated in the SCAN; those not visible
                      in the crop are supervised to be empty with weight neg_weight (0.5); unannotated are skipped.
    net_output: softmax probabilities (B, C, ...), target: one-hot (B, C, ...), annotated: (B, C) 0/1 (scan_level only)."""

    def __init__(self, num_classes, scan_level=False, neg_weight=0.5, history_window=50):
        super().__init__()
        from models.Proposed.sing_class_loss import CrossEntropyLoss3d
        from utils.losses import DiceLoss
        self.C, self.scan_level, self.neg_weight = num_classes, scan_level, neg_weight
        self.ce_crit = CrossEntropyLoss3d()
        self.dice_crit = DiceLoss()
        self.history = deque(maxlen=history_window)

    def forward(self, net_output, target, annotated=None):
        dev, C = net_output.device, self.C
        ce_loss = net_output.sum() * 0.0
        dice_loss = net_output.sum() * 0.0
        target_rev = torch.argmax(target, dim=1)
        tot, cnt = torch.zeros(C, device=dev), torch.zeros(C, device=dev)
        info = []
        for a in range(target.shape[0]):
            present = torch.unique(target_rev[a])
            present = present[present != 0]
            if self.scan_level:
                classes = [i for i in range(1, C) if float(annotated[a, i]) > 0.5]
            else:
                if present.nelement() == 0:
                    continue
                classes = [int(i) for i in present]
            for i in classes:
                target_class = target[a, i].unsqueeze(0).long()
                pre = net_output[a, [0, i]].clone()
                pre[0] = torch.clamp(1.0 - pre[1], min=1e-10, max=1.0)
                lam = 1.0 if (not self.scan_level or i in [int(p) for p in present]) else self.neg_weight
                ce = self.ce_crit(pre.unsqueeze(0), target_class)
                dice = self.dice_crit(pre.unsqueeze(0).permute(0, 2, 3, 4, 1), F.one_hot(target_class, 2), num_class=2)
                tot[i] += dice.detach()
                cnt[i] += 1
                info.append((ce, dice, i, lam))
        self.history.append(tot / (cnt + 1e-10))
        avg, n = torch.zeros(C, device=dev), torch.zeros(C, device=dev)
        for h in self.history:
            avg += h
            n += (h > 0).float()
        avg = torch.where(n > 0, avg / n.clamp(min=1.0), torch.zeros_like(avg))
        valid = (avg[1:] > 0).nonzero(as_tuple=True)[0] + 1
        mean_loss = avg[valid].mean() if valid.nelement() > 0 else 1.0
        w = torch.clamp(avg / mean_loss, min=0.5)
        for ce, dice, i, lam in info:
            ce_loss = ce_loss + ce * w[i] * lam
            dice_loss = dice_loss + dice * w[i] * lam
        return ce_loss + dice_loss


class PLSegLoss(nn.Module):
    """Total PL-Seg loss as in train_PLSeg.py: pyramid partial-label supervision (HADF at the 4 output sizes, averaged)
    + consistency_weight(epoch) * progressive self-distillation + alpha * class-wise orthogonal contrastive term."""

    def __init__(self, num_classes, scan_level=False, epochs=250, consistency=0.1, temperature=10.0, alpha=0.1):
        super().__init__()
        from utils.losses import DistillationLoss
        self.C, self.epochs, self.consistency, self.alpha = num_classes, epochs, consistency, alpha
        self.hadf = nn.ModuleList([HADFCMF(num_classes, scan_level) for _ in range(4)])
        self.distill = DistillationLoss(T=temperature)

    def forward(self, outs, label, annotated, epoch):
        from utils.losses import CO_Contrastive
        from utils.ramps import sigmoid_rampup
        o1, o2, o3, o4 = outs
        onehot = F.one_hot(label.long(), self.C).permute(0, 4, 1, 2, 3).float()
        soft = [torch.clamp(torch.softmax(o, dim=1), min=1e-10, max=1.0) for o in outs]
        sup = 0.0
        for k, o in enumerate(outs):
            t = onehot if k == 0 else F.interpolate(onehot, size=o.shape[2:], mode='nearest')
            sup = sup + self.hadf[k](soft[k], t, annotated)
        sup = sup / 4.0
        pseudo = 0.0
        for teacher, student in ((o1, o2), (o2, o3), (o3, o4)):
            tt = F.interpolate(teacher.detach(), size=student.shape[2:], mode='trilinear', align_corners=True)
            pseudo = pseudo + self.distill(student, tt)
        pseudo = pseudo / 3.0
        co = CO_Contrastive(soft[0])
        total = sup + self.consistency * sigmoid_rampup(epoch, self.epochs) * pseudo + self.alpha * co
        return total, {'sup': float(sup.detach()), 'distill': float(pseudo.detach()), 'cocr': float(co.detach())}


# Names: <method>_pl4mis = as in the PL4MIS toolbox, <method>_pl4cmf = our fixes for the CMF benchmark.
# mt_stock is reserved for the original MultiTalent repo (not ported yet).
_OLD_NAMES = {'me_stock': 'me_pl4mis', 'me_fixed': 'me_pl4cmf', 'mt_fixed': 'mt_pl4cmf',
              'plseg_stock': 'plseg_pl4mis', 'plseg_hybrid': 'plseg_pl4cmf', 'leaf': 'leaf_pl4cmf'}


def make_loss(name, num_classes, **kw):
    if name == 'mt_stock':
        raise ValueError("mt_stock is reserved for the original MultiTalent repo; "
                         "use mt_pl4mis (PL4MIS toolbox version) or mt_pl4cmf (our fix)")
    name = _OLD_NAMES.get(name, name)
    if name in ('plseg_pl4mis', 'plseg_pl4cmf'):
        return PLSegLoss(num_classes, scan_level=(name == 'plseg_pl4cmf'), **kw)
    if name == 'mt_pl4mis':
        return MultiTalentStock()
    if name == 'mt_pl4cmf':
        return MultiTalentCMF()
    if name == 'me_pl4mis':
        return MarginalExclusionCMF(num_classes, 'crop', 'logits')
    if name == 'me_pl4cmf':
        return MarginalExclusionCMF(num_classes, 'case', 'probs')
    if name == 'leaf_pl4cmf':
        return LeafDicePL4CMF(num_classes)
    if name == 'leaf_merged':
        return LeafDiceCMF(num_classes)
    raise ValueError(name)
