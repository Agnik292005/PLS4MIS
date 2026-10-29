import sys, time
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from train_DoDNet_cmf import Compose
from dataloader.cmf_case_dataset import CMFCaseDataset
from dataloader.cmf_transforms import ZScore, PadToPatch, ToTensorZ
from dataloader.transforms import LabeledClass, TrainerCrop, CreateOnehotLabel

P = [192, 192, 192]
tf = Compose([ZScore(), PadToPatch(P), LabeledClass(7), TrainerCrop(P), CreateOnehotLabel(7), ToTensorZ()])
ds = CMFCaseDataset(sys.argv[1], 'train', tf, 7, 42)
nw = int(sys.argv[2])
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=nw, drop_last=True)
t0 = time.time()
for i, b in enumerate(dl):
    if i == 4:
        t1 = time.time()
    if i == 44:
        break
print('workers=%d: first batch after warmup %.0f s, then %.2f s per sample' % (nw, t1 - t0, (time.time() - t1) / 40))
