"""
train_unet_cmf.py

U-Net ("lower bound") training for the CMF benchmark. New file; the stock
train_unet.py is not touched. Same protocol as train_DoDNet_cmf.py:
  * NumPy/b2nd cache, whole-volume z-score, short scans padded to the patch
  * batch 1, 250 epochs, poly learning-rate decay (--lr_schedule plateau gives
    the stock behaviour), optional gradient clipping, resumable (latest.pth)
  * annotation-aware validation (cmf_eval), background not scored, best
    checkpoint by the mean of the six per-class means, final-epoch model is
    final_model.pth and gets the full validation
What is specific to the U-Net:
  * one softmax over all classes; a structure that is not labelled in a scan
    is background in that scan's label (the cache stores exactly that)
  * --loss ce_dice (default): 0.5 * (soft Dice over all classes + cross-entropy), as in the stock script
  * --loss me_pl4mis | me_pl4cmf | leaf_pl4cmf | mt_pl4mis | mt_pl4cmf | plseg_pl4mis | plseg_pl4cmf: partial-label losses, see models/cmf_partial_losses.py (old names me_stock, me_fixed, leaf, mt_fixed, plseg_stock, plseg_hybrid still work)
    (mt_*: one sigmoid per structure, so prediction uses decode='sigmoid': background fixed at 0.5, then argmax)
  * Adam (0.9, 0.99), Xavier init, BatchNorm network (models/unet3d.py), as stock
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader.cmf_cache_dataset import CMFCacheDataset, EpochShuffleSampler
from models.unet3d import UNet3D
from models.weight_init import initialize_weights
from utils.losses import DiceLoss
from models.cmf_partial_losses import make_loss
from utils.cmf_eval import per_class_dice, ClassMeter, FOREGROUND
from utils.cmf_val_softmax import predict_volume_softmax


class CMFPartialDataset(CMFCacheDataset):
    """Adds 'annotated' (num_classes,) = classes present in the WHOLE scan label (cache metadata), to training items."""

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if self.mode == 'train':
            index = key[0] if isinstance(key, tuple) else key
            ann = torch.zeros(self.num_classes, dtype=torch.float32)
            for c in self.rows[index]['present']:
                if 0 < c < self.num_classes:
                    ann[c] = 1.0
            item['annotated'] = ann
        return item


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', required=True, help='NumPy/b2nd cache from data_prep/make_cmf_cache.py')
    p.add_argument('--snapshot_dir', required=True)
    p.add_argument('--patch_size', type=int, nargs='+', required=True, help='x y z')
    p.add_argument('--num_classes', type=int, default=7)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--epoches', type=int, default=250)
    p.add_argument('--optim', choices=['adam', 'sgd'], default='adam')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lr_schedule', choices=['poly', 'plateau'], default='poly')
    p.add_argument('--clip_grad', type=float, default=0.0, help='clip gradient norm at this value (0 = off, only log it)')
    p.add_argument('--loss', choices=['ce_dice', 'me_pl4mis', 'me_pl4cmf', 'leaf_pl4cmf', 'leaf_merged', 'mt_pl4mis', 'mt_pl4cmf', 'plseg_pl4mis', 'plseg_pl4cmf', 'me_stock', 'me_fixed', 'leaf', 'mt_fixed', 'plseg_stock', 'plseg_hybrid'], default='ce_dice',
                   help='ce_dice = plain U-Net (unannotated = background); me_* = Marginal Exclusion loss; leaf = LeafDice idea; mt_* = MultiTalent loss (sigmoid outputs)')
    p.add_argument('--weight_init', default='xavier')
    p.add_argument('--consistency', type=float, default=0.1, help='PL-Seg: weight of the self-distillation term')
    p.add_argument('--temperature', type=float, default=10.0, help='PL-Seg: distillation temperature')
    p.add_argument('--alpha', type=float, default=0.1, help='PL-Seg: weight of the contrastive term')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val_interval', type=int, default=25, help='epochs between in-training validations; 0 = none')
    p.add_argument('--val_subset', type=int, default=20, help='cases used in in-training validation; 0 = all')
    p.add_argument('--val_overlap', type=float, default=0.0)
    p.add_argument('--final_val_overlap', type=float, default=0.5)
    p.add_argument('--save_interval', type=int, default=5, help='epochs between latest.pth saves')
    p.add_argument('--max_iters', type=int, default=0, help='stop each epoch after this many iterations (smoke tests)')
    p.add_argument('--resume', default='', help='path to latest.pth')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_ckpt(path, model, optim, epoch, best):
    tmp = path + '.tmp'
    torch.save({'model': model.state_dict(), 'optim': optim.state_dict(), 'epoch': epoch, 'best': best}, tmp)
    os.replace(tmp, path)


def subset_indices(n, k):
    if k <= 0 or k >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)] if k > 1 else [0]


def fmt(summary):
    pc = ' '.join('c%d=%.3f(n=%d)' % (c, summary['per_class'][c], summary['n_cases'][c]) for c in summary['per_class'])
    return '%s | anatomy=%.4f lesion=%.4f mean6=%.4f' % (pc, summary['anatomy_mean'], summary['lesion'], summary['mean_all'])


def validate(model, val_ds, args, overlap, indices=None):
    patch_zyx = tuple(args.patch_size[::-1])
    meter = ClassMeter(FOREGROUND)
    idx = range(len(val_ds)) if indices is None else indices
    for i in idx:
        s = val_ds[i]
        gt = s['label'].numpy().astype(np.int64)
        decode = 'sigmoid' if args.loss.startswith('mt') else 'softmax'
        pred = predict_volume_softmax(model, s['image'][0], args.num_classes, patch_zyx, overlap, args.device, decode)
        meter.update(per_class_dice(pred, gt, FOREGROUND))
    return meter.summary()


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.snapshot_dir, exist_ok=True)
    seed_all(args.seed)
    dev = torch.device(args.device)
    nc = args.num_classes

    train_ds = CMFPartialDataset(args.cache_dir, 'train', args.patch_size, nc, args.seed)
    val_ds = CMFPartialDataset(args.cache_dir, 'val', args.patch_size, nc, args.seed)
    sampler = EpochShuffleSampler(len(train_ds), args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        drop_last=True, pin_memory=(dev.type == 'cuda'),
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)

    params = {'in_chns': 1, 'class_num': nc, 'feature_chns': [16, 32, 64, 128],
              'dropout': [0, 0, 0.1, 0.2], 'trilinear': True, 'multiscale_pred': False}
    if args.loss.startswith('plseg'):                      # PL-Seg: same U-Net body, four outputs (repo file, unchanged)
        from models.Proposed.unet3d_ProgCon import UNet3D_ProgCon
        model = UNet3D_ProgCon(params).to(dev)
    else:
        model = UNet3D(params).to(dev)
    print('parameters:', sum(p.numel() for p in model.parameters()), flush=True)
    start_epoch, best = 0, -1.0
    if not args.resume:
        initialize_weights(model, args.weight_init)
    if args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True)
    plateau = None
    if args.lr_schedule == 'plateau':
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.95, patience=4, min_lr=1e-4)
    dice_crit = DiceLoss()
    kw = dict(epochs=args.epoches, consistency=args.consistency, temperature=args.temperature, alpha=args.alpha) \
        if args.loss.startswith('plseg') else {}
    crit = None if args.loss == 'ce_dice' else make_loss(args.loss, nc, **kw)

    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck['model'])
        optim.load_state_dict(ck['optim'])
        start_epoch, best = int(ck['epoch']) + 1, float(ck['best'])
        print('resumed from %s: next epoch %d, best %.4f' % (args.resume, start_epoch, best))

    log_path = os.path.join(args.snapshot_dir, 'task_log.csv')
    new_log = not os.path.exists(log_path)
    logf = open(log_path, 'a')
    if new_log:
        logf.write('epoch,case,classes_in_crop,loss\n')

    for epoch in range(start_epoch, args.epoches):
        if plateau is None:
            for g in optim.param_groups:
                g['lr'] = args.lr * (1.0 - epoch / float(args.epoches)) ** 0.9
        sampler.set_epoch(epoch)
        model.train()
        t0, tot, nit = time.time(), 0.0, 0
        part_tot = {}
        for it, b in enumerate(loader):
            if args.max_iters and it >= args.max_iters:
                break
            image = b['image'].to(dev)
            lab = b['label'].to(dev).long()                                   # (B, D, H, W), 0..nc-1
            onehot = F.one_hot(lab, nc).permute(0, 4, 1, 2, 3).float()        # (B, nc, D, H, W)
            logits = model(image)
            if crit is None:
                soft = torch.clamp(torch.softmax(logits, dim=1), min=1e-10, max=1.0)
                d_loss = dice_crit(soft.permute(0, 2, 3, 4, 1), onehot.permute(0, 2, 3, 4, 1), num_class=nc)
                ce_loss = F.cross_entropy(logits, lab)
                loss = 0.5 * (d_loss + ce_loss)
            elif args.loss.startswith('plseg'):
                loss, parts = crit(logits, lab, b['annotated'].to(dev), epoch)
                for k_, v_ in parts.items():
                    part_tot[k_] = part_tot.get(k_, 0.0) + v_
            else:
                loss = crit(logits, lab, b['annotated'].to(dev))
            optim.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad if args.clip_grad > 0 else float('inf'))
            with open(os.path.join(args.snapshot_dir, 'grad_norm.csv'), 'a') as _gf:
                _gf.write('%.3f\n' % float(gn))
            optim.step()
            tot += float(loss.item()); nit += 1
            counts = torch.bincount(lab.flatten(), minlength=nc).tolist()
            inside = '|'.join(str(c) for c in range(1, nc) if counts[c] > 0) or '-'
            for k in range(image.shape[0]):
                logf.write('%d,%s,%s,%.4f\n' % (epoch, b['img_name'][k], inside, float(loss.item())))
        logf.flush()
        if part_tot:
            print('  parts: ' + ' '.join('%s=%.4f' % (k_, v_ / max(nit, 1)) for k_, v_ in part_tot.items()), flush=True)
        print('epoch %d/%d  loss %.4f  lr %.5f  %.0fs' % (
            epoch + 1, args.epoches, tot / max(nit, 1), optim.param_groups[0]['lr'], time.time() - t0), flush=True)
        if args.device.startswith('cuda'):
            print('  peak GPU memory %.1f GB' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)

        done = (epoch + 1 == args.epoches)
        if args.val_interval and (epoch + 1) % args.val_interval == 0 and not done:
            s = validate(model, val_ds, args, args.val_overlap, subset_indices(len(val_ds), args.val_subset))
            print('  val(subset) ' + fmt(s), flush=True)
            if plateau is not None:
                plateau.step(s['mean_all'])
            if s['mean_all'] > best:
                best = s['mean_all']
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'best_model.pth'))
                print('  new best %.4f' % best, flush=True)
        if (epoch + 1) % args.save_interval == 0 or done:
            save_ckpt(os.path.join(args.snapshot_dir, 'latest.pth'), model, optim, epoch, best)

    save_ckpt(os.path.join(args.snapshot_dir, 'final_model.pth'), model, optim, args.epoches - 1, best)
    s = validate(model, val_ds, args, args.final_val_overlap)
    print('FINAL full validation ' + fmt(s), flush=True)
    with open(os.path.join(args.snapshot_dir, 'final_val.json'), 'w') as f:
        json.dump(s, f, indent=2)
    logf.close()
    return s


if __name__ == '__main__':
    main()
