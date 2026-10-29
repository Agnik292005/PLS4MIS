"""
cmf_val_softmax.py

Sliding-window inference for ordinary softmax networks (U-Net, MELoss,
LeafDice, ...) on the CMF data. Same tiling rules as utils/cmf_val.py
(predict_volume, used by DoDNet): short scans padded with 0 (scan mean after
z-score), tiles may overlap and their probabilities are averaged, the result
is cropped back to the original shape. Array order is (z, y, x).
One forward pass per tile (the network outputs all classes at once).
"""
import numpy as np
import torch

from utils.cmf_val import _starts


@torch.no_grad()
def predict_volume_softmax(model, image, num_classes, patch_zyx, overlap=0.0, device="cuda", decode="softmax"):
    image = torch.as_tensor(np.asarray(image), dtype=torch.float32)
    orig = tuple(image.shape)
    pads = []
    for d, p in zip(orig, patch_zyx):
        deficit = max(p - d, 0)
        pads.append((deficit // 2, deficit - deficit // 2))
    if any(p != (0, 0) for p in pads):
        flat = [v for pr in pads[::-1] for v in pr]
        image = torch.nn.functional.pad(image, flat, mode='constant', value=0.0)
    Z, Y, X = image.shape
    strides = [max(1, int(round(p * (1.0 - overlap)))) for p in patch_zyx]
    prob = torch.zeros((num_classes, Z, Y, X), dtype=torch.float32)
    cnt = torch.zeros((Z, Y, X), dtype=torch.float32)
    model.eval()
    for zs in _starts(Z, patch_zyx[0], strides[0]):
        for ys in _starts(Y, patch_zyx[1], strides[1]):
            for xs in _starts(X, patch_zyx[2], strides[2]):
                sl = (slice(zs, zs + patch_zyx[0]), slice(ys, ys + patch_zyx[1]), slice(xs, xs + patch_zyx[2]))
                tile = image[sl].to(device)[None, None]
                out = model(tile)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if decode == "softmax":
                    pt = torch.softmax(out[0].float(), dim=0).cpu()
                else:  # "sigmoid": one yes/no output per structure, background fixed at 0.5 (as in DoDNet)
                    pt = torch.sigmoid(out[0].float()).cpu()
                    pt[0] = 0.5
                prob[(slice(None),) + sl] += pt
                cnt[sl] += 1.0
    prob /= cnt.clamp(min=1.0)
    lab = prob.argmax(0).numpy().astype(np.int16)
    sl = tuple(slice(b, b + o) for (b, _), o in zip(pads, orig))
    return lab[sl]
