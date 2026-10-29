import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import train_DoDNet_cmf as T
from dataloader.cmf_cache_dataset import CMFCacheDataset

CACHE = '/DATA/scratch/sukhraj/PLS4MIS_CMF_cache'
RUN = '/DATA/scratch/sukhraj/runs/dodnet_cmf_seed42'
argv = ['--cache_dir', CACHE, '--snapshot_dir', '/tmp/head_check',
        '--patch_size', '192', '192', '192', '--weight_std'] + sys.argv[1:]
args = T.get_args(argv)
dev = torch.device(args.device)
val_ds = CMFCacheDataset(args.cache_dir, 'val', args.patch_size,
                         num_classes=args.num_classes, seed=args.seed)
idx = T.subset_indices(len(val_ds), 20)
stats = {}
shown = []

def hook(mod, a, kw, out):
    cands = [t for t in list(a[1:]) + list(kw.values())
             if torch.is_tensor(t) and t.numel() <= 64]
    if not shown:
        shown.append(1)
        print('hook sees args:', [tuple(t.shape) if torch.is_tensor(t) else type(t).__name__ for t in a],
              'kwargs:', list(kw.keys()), 'candidates:', [tuple(t.shape) for t in cands])
    if not cands:
        return
    c = int(cands[0].flatten().argmax()) + 1
    o = out[0] if isinstance(out, (tuple, list)) else out
    o = o.detach().float()
    s = stats.setdefault(c, dict(n=0, sm=0.0, sq=0.0, mx=-1e9, mn=1e9, pos=0))
    s['n'] += o.numel()
    s['sm'] += o.sum().item()
    s['sq'] += (o * o).sum().item()
    s['mx'] = max(s['mx'], o.max().item())
    s['mn'] = min(s['mn'], o.min().item())
    s['pos'] += (o > 0).sum().item()

model = T.UNet3D(weight_std=args.weight_std, num_classes=args.num_classes).to(dev)
ck = torch.load(os.path.join(RUN, 'final_model.pth'), map_location='cpu', weights_only=False)
model.load_state_dict(ck['model'] if 'model' in ck else ck)
model.eval()
h = model.register_forward_hook(hook, with_kwargs=True)
with torch.no_grad():
    summ = T.validate(model, val_ds, args, 0.0, indices=idx)
h.remove()
print('\n== final_model.pth, 20 cases')
print(T.fmt(summ))
for c in sorted(stats):
    s = stats[c]
    mean = s['sm'] / s['n']
    std = math.sqrt(max(s['sq'] / s['n'] - mean * mean, 0.0))
    print('task %d  mean %.2f  std %.2f  min %.2f  max %.2f  share>0 %.5f'
          % (c, mean, std, s['mn'], s['mx'], s['pos'] / s['n']))
