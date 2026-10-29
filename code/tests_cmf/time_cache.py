"""Times the new cache loader: python tests_cmf/time_cache.py <cache_dir> <workers> [patch]"""
import sys, time
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from dataloader.cmf_cache_dataset import CMFCacheDataset, EpochShuffleSampler

cache, nw = sys.argv[1], int(sys.argv[2])
patch = int(sys.argv[3]) if len(sys.argv) > 3 else 192
ds = CMFCacheDataset(cache, 'train', (patch, patch, patch), 7, 42)
sampler = EpochShuffleSampler(len(ds), 42)
dl = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=nw, drop_last=True,
                persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
n_warm, n_meas = 4, 40
t0, t1, empty, i, epoch, done = time.time(), None, 0, 0, 0, False
while not done:
    sampler.set_epoch(epoch)
    for b in dl:
        if i == n_warm:
            t1 = time.time()
        if i >= n_warm:
            empty += int((b['label'] == b['forced_class'].view(-1, 1, 1, 1)).sum() == 0)
        i += 1
        if i == n_warm + n_meas:
            done = True
            break
    epoch += 1
print('workers=%d: warmup %.0f s, then %.2f s per sample; crops without the chosen class %d/%d'
      % (nw, t1 - t0, (time.time() - t1) / n_meas, empty, n_meas))
