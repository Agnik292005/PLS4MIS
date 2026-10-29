"""CPU smoke test: tiny fake dataset, tiny patch, 2 epochs, then resume."""
import os, tempfile, unittest
import numpy as np
import SimpleITK as sitk
import train_DoDNet_cmf as T


def _make(root, n_tr=3, n_val=2):
    rng = np.random.RandomState(0)
    for split, n, il, ll in (('Tr', n_tr, 'imagesTr', 'labelsTr'), ('Val', n_val, 'imagesVal', 'labelsVal')):
        os.makedirs(os.path.join(root, il)); os.makedirs(os.path.join(root, ll))
        for i in range(n):
            shape = (20 + 4 * i, 40, 40)  # z shorter than the 32 patch for case 0
            img = rng.randn(*shape).astype(np.float32)
            lab = np.zeros(shape, np.int16)
            lab[:, 5:15, 5:15] = 1
            lab[:, 20:30, 20:30] = 2 + (i % 2)
            if i == 2: lab[:, 30:35, 5:10] = 6
            sitk.WriteImage(sitk.GetImageFromArray(img), os.path.join(root, il, 'c%d_0000.nii.gz' % i))
            sitk.WriteImage(sitk.GetImageFromArray(lab), os.path.join(root, ll, 'c%d.nii.gz' % i))


class Smoke(unittest.TestCase):
    def test_train_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            data, snap = os.path.join(d, 'data'), os.path.join(d, 'snap')
            _make(data)
            base = ['--data_dir', data, '--snapshot_dir', snap, '--patch_size', '32', '32', '32',
                    '--device', 'cpu', '--workers', '0', '--val_interval', '1', '--save_interval', '1',
                    '--final_val_overlap', '0.0']
            s = T.main(base + ['--epoches', '2'])
            self.assertTrue(os.path.exists(os.path.join(snap, 'latest.pth')))
            self.assertTrue(os.path.exists(os.path.join(snap, 'final_val.json')))
            self.assertEqual(set(s['per_class']), {1, 2, 3, 4, 5, 6})
            T.main(base + ['--epoches', '3', '--resume', os.path.join(snap, 'latest.pth')])
            with open(os.path.join(snap, 'task_log.csv')) as f:
                lines = f.read().strip().split('\n')
            epochs = sorted({int(l.split(',')[0]) for l in lines[1:]})
            self.assertEqual(epochs, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
