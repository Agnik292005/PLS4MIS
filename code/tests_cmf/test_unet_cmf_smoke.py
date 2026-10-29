"""CPU smoke test for train_unet_cmf.py: tiny fake cache, tiny patch, 2 epochs, then resume."""
import json, os, tempfile, unittest
import numpy as np
import train_unet_cmf as U


def _make_cache(root, n_tr=3, n_val=2):
    rng = np.random.RandomState(0)
    for split, n in (('train', n_tr), ('val', n_val)):
        os.makedirs(os.path.join(root, split))
        for i in range(n):
            shape = (20 + 4 * i, 40, 40)          # case 0 is shorter than the 32 patch in z
            img = rng.randn(*shape).astype(np.float32)
            lab = np.zeros(shape, np.uint8)
            lab[:, 5:15, 5:15] = 1
            lab[:, 20:30, 20:30] = 2 + (i % 2)
            if i == 2:
                lab[:, 30:35, 5:10] = 6
            base = os.path.join(root, split, 'c%d' % i)
            np.save(base + '_img.npy', img)
            np.save(base + '_lab.npy', lab)
            with open(base + '_meta.json', 'w') as f:
                json.dump({'present': [int(c) for c in np.unique(lab) if c > 0]}, f)


class Smoke(unittest.TestCase):
    def test_train_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            cache, snap = os.path.join(d, 'cache'), os.path.join(d, 'snap')
            _make_cache(cache)
            base = ['--cache_dir', cache, '--snapshot_dir', snap, '--patch_size', '32', '32', '32',
                    '--device', 'cpu', '--workers', '0', '--val_interval', '1', '--save_interval', '1',
                    '--final_val_overlap', '0.0', '--clip_grad', '12']
            s = U.main(base + ['--epoches', '2'])
            for f in ('latest.pth', 'final_model.pth', 'final_val.json', 'grad_norm.csv', 'task_log.csv'):
                self.assertTrue(os.path.exists(os.path.join(snap, f)), f)
            self.assertEqual(set(s['per_class']), {1, 2, 3, 4, 5, 6})
            U.main(base + ['--epoches', '3', '--resume', os.path.join(snap, 'latest.pth')])
            with open(os.path.join(snap, 'task_log.csv')) as f:
                lines = f.read().strip().split('\n')
            self.assertEqual(sorted({int(l.split(',')[0]) for l in lines[1:]}), [0, 1, 2])

    def test_every_loss_trains(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, 'cache')
            _make_cache(cache)
            for name in ('ce_dice', 'me_stock', 'me_fixed', 'leaf', 'mt_stock', 'mt_fixed', 'plseg_stock', 'plseg_hybrid'):
                snap = os.path.join(d, 'snap_' + name)
                s = U.main(['--cache_dir', cache, '--snapshot_dir', snap, '--patch_size', '32', '32', '32',
                            '--device', 'cpu', '--workers', '0', '--val_interval', '0', '--epoches', '2',
                            '--final_val_overlap', '0.0', '--loss', name])
                self.assertEqual(set(s['per_class']), {1, 2, 3, 4, 5, 6}, name)
                self.assertTrue(os.path.exists(os.path.join(snap, 'final_model.pth')), name)


if __name__ == '__main__':
    unittest.main()
