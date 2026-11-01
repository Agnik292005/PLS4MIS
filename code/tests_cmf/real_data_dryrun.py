"""
Real-cache dry run (CPU, no training, no validation). For a few REAL training crops it runs, for every --loss variant,
model -> loss -> backward and checks that everything is finite and that gradients exist. It also checks that the
stored list of annotated structures ('present' in the cache metadata) agrees with the label inside each crop.
Usage:  python tests_cmf/real_data_dryrun.py --cache_dir /path/to/PLS4MIS_CMF_cache --patch 64 --cases 6
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import train_unet_cmf as U
from models.unet3d import UNet3D
from models.Proposed.unet3d_ProgCon import UNet3D_ProgCon
from models.cmf_partial_losses import make_loss
from utils.losses import DiceLoss

VARIANTS = ['ce_dice', 'me_pl4mis', 'me_pl4cmf', 'leaf_pl4cmf', 'mt_pl4mis', 'mt_pl4cmf', 'plseg_pl4mis', 'plseg_pl4cmf']


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--patch', type=int, default=64, help='cube edge in voxels (192 is the real size; 64 is fast on a CPU)')
    p.add_argument('--cases', type=int, default=6)
    a = p.parse_args(argv)
    torch.manual_seed(0)
    nc = 7
    ds = U.CMFPartialDataset(a.cache_dir, 'train', [a.patch] * 3, nc, 42)
    idx = sorted(set(np.linspace(0, len(ds) - 1, min(a.cases, len(ds))).astype(int).tolist()))
    items = [ds[(i, 0)] for i in idx]
    problems = 0

    print('--- data check (%d real crops, %d^3) ---' % (len(items), a.patch))
    for it in items:
        lab = it['label'].numpy().astype(int)
        ann = it['annotated'].numpy()
        in_crop = sorted(set(np.unique(lab).tolist()) - {0})
        not_annotated = [c for c in in_crop if ann[c] < 0.5]
        ok = (lab.min() >= 0 and lab.max() < nc and not not_annotated and np.isfinite(it['image'].numpy()).all())
        problems += 0 if ok else 1
        print('%-28s classes in crop %-14s annotated in scan %-18s %s' % (
            it['img_name'], in_crop, [c for c in range(1, nc) if ann[c] > 0.5], 'ok' if ok else 'PROBLEM %s' % not_annotated))

    params = {'in_chns': 1, 'class_num': nc, 'feature_chns': [16, 32, 64, 128],
              'dropout': [0, 0, 0.1, 0.2], 'trilinear': True, 'multiscale_pred': False}
    print('--- loss check (one forward/backward per crop) ---')
    for name in VARIANTS:
        t0 = time.time()
        torch.manual_seed(0)
        model = (UNet3D_ProgCon if name.startswith('plseg') else UNet3D)(params)
        crit = None if name == 'ce_dice' else make_loss(name, nc, **(dict(epochs=250) if name.startswith('plseg') else {}))
        losses, fine = [], True
        for it in items:
            model.zero_grad()
            image, lab, ann = it['image'][None], it['label'][None].long(), it['annotated'][None]
            out = model(image)
            if name == 'ce_dice':                      # exactly the loss line of train_unet_cmf.py
                onehot = F.one_hot(lab, nc).permute(0, 4, 1, 2, 3).float()
                soft = torch.clamp(torch.softmax(out, dim=1), min=1e-10, max=1.0)
                d_loss = DiceLoss()(soft.permute(0, 2, 3, 4, 1), onehot.permute(0, 2, 3, 4, 1), num_class=nc)
                loss = 0.5 * (d_loss + F.cross_entropy(out, lab))
                repo_style = 0.5 * (d_loss + torch.nn.CrossEntropyLoss()(out, onehot))   # as written in the repo's train_unet.py
                if abs(float(repo_style) - float(loss)) > 1e-5:
                    print('   MISMATCH with the repo-style U-Net loss:', float(loss), float(repo_style))
                    problems += 1
            elif name.startswith('plseg'):
                loss, _ = crit(out, lab, ann, 0)
            else:
                loss = crit(out, lab, ann)
            loss.backward()
            g = [q.grad for q in model.parameters() if q.grad is not None]
            empty = int(lab.max()) == 0
            need_grad = not (name in ('me_pl4mis', 'mt_pl4mis', 'leaf_pl4cmf') and empty)   # no labelled voxel in the crop: these three have nothing to learn from   # stock losses grade nothing on an empty crop
            fine &= bool(torch.isfinite(loss)) and all(torch.isfinite(x).all() for x in g) and (sum(float(x.abs().sum()) for x in g) > 0 or not need_grad)
            losses.append(float(loss))
        problems += 0 if fine else 1
        print('%-14s %-7s loss per crop: %s   (%.0fs)' % (name, 'ok' if fine else 'PROBLEM', ' '.join('%.3f' % l for l in losses), time.time() - t0))
    print('DRY RUN %s' % ('PASSED' if problems == 0 else 'FOUND %d PROBLEM(S)' % problems))
    return problems


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
