"""
train_DoDNet_cmf.py

DoDNet training for the CMF benchmark (PLAN.md). New file; the stock
train_DoDNet.py is not touched. What differs from the stock script:
  * one row per case, one task per visit (shuffled cycle, cmf_task_select)
  * whole-volume z-score, short scans padded to the patch (no crash)
  * annotation-aware validation (cmf_eval), background not scored,
    best checkpoint by the mean of the six per-class means
  * validation pads short scans and can overlap tiles (cmf_val)
  * latest.pth saved during training; --resume restores model, optimiser,
    epoch and best score
  * learning rate decays with a poly schedule by default (does not depend on
    how often we validate); --lr_schedule plateau gives the stock behaviour
  * logs which task was trained and whether the crop contained that class
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader.cmf_case_dataset import CMFCaseDataset
from dataloader.cmf_cache_dataset import CMFCacheDataset, EpochShuffleSampler
from dataloader.cmf_transforms import ZScore, PadToPatch, ToTensorZ, Test_ToTensorZ
from dataloader.transforms import LabeledClass, TrainerCrop, CreateOnehotLabel
from models.DoDNet.unet3D_DynConv882 import UNet3D
from models.DoDNet.loss import DiceLoss4MOTS, CELoss4MOTS
from utils.cmf_eval import per_class_dice, ClassMeter, FOREGROUND
from utils.cmf_val import predict_volume


class Compose(object):
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, sample):
        for f in self.fns:
            sample = f(sample)
        return sample


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='', help='nii.gz folder (slow path)')
    p.add_argument('--cache_dir', default='', help='NumPy cache from data_prep/make_cmf_cache.py (fast path)')
    p.add_argument('--snapshot_dir', required=True)
    p.add_argument('--patch_size', type=int, nargs='+', required=True, help='x y z')
    p.add_argument('--num_classes', type=int, default=7)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--epoches', type=int, default=250)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lr_schedule', choices=['poly', 'plateau'], default='poly')
    p.add_argument('--clip_grad', type=float, default=0.0, help='clip gradient norm at this value (0 = off, only log it)')
    p.add_argument('--weight_std', action='store_true')
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


def save_ckpt(path, model, optim, epoch, best, extra=None):
    d = {'model': model.state_dict(), 'optim': optim.state_dict(), 'epoch': epoch, 'best': best}
    if extra:
        d.update(extra)
    tmp = path + '.tmp'
    torch.save(d, tmp)
    os.replace(tmp, path)


def validate(model, val_ds, args, overlap, indices=None):
    patch_zyx = tuple(args.patch_size[::-1])
    meter = ClassMeter(FOREGROUND)
    idx = range(len(val_ds)) if indices is None else indices
    for i in idx:
        s = val_ds[i]
        img = s['image'][0]
        gt = s['label'].numpy().astype(np.int64)
        pred = predict_volume(model, img, args.num_classes, patch_zyx, overlap, args.device)
        meter.update(per_class_dice(pred, gt, FOREGROUND))
    return meter.summary()


def subset_indices(n, k):
    if k <= 0 or k >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)] if k > 1 else [0]


def fmt(summary):
    pc = ' '.join('c%d=%.3f(n=%d)' % (c, summary['per_class'][c], summary['n_cases'][c]) for c in summary['per_class'])
    return '%s | anatomy=%.4f lesion=%.4f mean6=%.4f' % (pc, summary['anatomy_mean'], summary['lesion'], summary['mean_all'])


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.snapshot_dir, exist_ok=True)
    seed_all(args.seed)
    dev = torch.device(args.device)
    nfg = args.num_classes - 1

    sampler = None
    if args.cache_dir:
        train_ds = CMFCacheDataset(args.cache_dir, 'train', args.patch_size, args.num_classes, args.seed)
        val_ds = CMFCacheDataset(args.cache_dir, 'val', args.patch_size, args.num_classes, args.seed)
        sampler = EpochShuffleSampler(len(train_ds), args.seed)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                            drop_last=True, pin_memory=(dev.type == 'cuda'),
                            persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    else:
        train_tf = Compose([ZScore(), PadToPatch(args.patch_size), LabeledClass(args.num_classes),
                            TrainerCrop(args.patch_size), CreateOnehotLabel(args.num_classes), ToTensorZ()])
        val_tf = Compose([ZScore(), Test_ToTensorZ()])
        train_ds = CMFCaseDataset(args.data_dir, 'train', train_tf, args.num_classes, args.seed)
        val_ds = CMFCaseDataset(args.data_dir, 'val', val_tf, args.num_classes, args.seed)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            drop_last=True, pin_memory=(dev.type == 'cuda'))

    model = UNet3D(weight_std=args.weight_std, num_classes=args.num_classes).to(dev)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True)
    plateau = None
    if args.lr_schedule == 'plateau':
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.95, patience=4, min_lr=1e-4)
    loss_dice = DiceLoss4MOTS(num_classes=1)
    loss_ce = CELoss4MOTS(num_classes=1, ignore_index=255)

    start_epoch, best = 0, -1.0
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
        logf.write('epoch,case,task,positive_voxels,loss\n')

    for epoch in range(start_epoch, args.epoches):
        if plateau is None:
            for g in optim.param_groups:
                g['lr'] = args.lr * (1.0 - epoch / float(args.epoches)) ** 0.9
        if sampler is not None:
            sampler.set_epoch(epoch)
        else:
            train_ds.set_epoch(epoch)
        model.train()
        t0, tot, nit, empty = time.time(), 0.0, 0, 0
        for it, b in enumerate(loader):
            if args.max_iters and it >= args.max_iters:
                break
            image = b['image'].to(dev)
            idx = (b['forced_class'].long() - 1).view(-1, 1).to(dev)
            enc = torch.zeros((image.shape[0], nfg), device=dev).scatter_(1, idx, 1.0)
            gt = (b['label'].to(dev) == b['forced_class'].to(dev).view(-1, 1, 1, 1)).float().unsqueeze(1)
            pos = gt.sum(dim=(1, 2, 3, 4))
            pred = model(image, enc)
            loss = loss_dice(pred, gt) + loss_ce(pred, gt)
            optim.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad if args.clip_grad > 0 else float('inf'))
            with open(os.path.join(args.snapshot_dir, 'grad_norm.csv'), 'a') as _gf:
                _gf.write('%.3f\n' % float(gn))
            optim.step()
            tot += float(loss.item()); nit += 1
            empty += int((pos == 0).sum().item())
            for k in range(image.shape[0]):
                logf.write('%d,%s,%d,%d,%.4f\n' % (epoch, b['img_name'][k], int(b['forced_class'][k]), int(pos[k].item()), float(loss.item())))
        logf.flush()
        print('epoch %d/%d  loss %.4f  lr %.5f  crops without the chosen class %d/%d  %.0fs' % (
            epoch + 1, args.epoches, tot / max(nit, 1), optim.param_groups[0]['lr'], empty, nit, time.time() - t0), flush=True)
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
