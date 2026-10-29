import sys, time
import torch
sys.path.insert(0, '.')
from train_DoDNet_cmf import Compose
from dataloader.cmf_case_dataset import CMFCaseDataset
from dataloader.cmf_transforms import ZScore, PadToPatch, ToTensorZ
from dataloader.transforms import LabeledClass, TrainerCrop, CreateOnehotLabel
from models.DoDNet.unet3D_DynConv882 import UNet3D
from models.DoDNet.loss import DiceLoss4MOTS, CELoss4MOTS

P = [192, 192, 192]
tf = Compose([ZScore(), PadToPatch(P), LabeledClass(7), TrainerCrop(P), CreateOnehotLabel(7), ToTensorZ()])
ds = CMFCaseDataset(sys.argv[1], 'train', tf, 7, 42)
ds.transform = None
t = time.time()
for i in range(0, 400, 40):
    ds[i]
print('read file only: %.1f s per case' % ((time.time() - t) / 10), flush=True)
ds.transform = tf
t = time.time()
for i in range(0, 400, 40):
    ds[i]
print('read + all transforms: %.1f s per case' % ((time.time() - t) / 10), flush=True)

dev = 'cuda'
m = UNet3D(weight_std=True, num_classes=7).to(dev)
opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.99, nesterov=True)
ld, lc = DiceLoss4MOTS(num_classes=1), CELoss4MOTS(num_classes=1, ignore_index=255)
x = torch.randn(1, 1, 192, 192, 192, device=dev)
gt = (torch.rand(1, 1, 192, 192, 192, device=dev) > 0.9).float()
for k in range(7):
    if k == 2:
        torch.cuda.synchronize(); t = time.time()
    enc = torch.zeros(1, 6, device=dev); enc[0, 0] = 1
    p = m(x, enc)
    loss = ld(p, gt) + lc(p, gt)
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize()
print('GPU step only: %.2f s per step' % ((time.time() - t) / 5))
