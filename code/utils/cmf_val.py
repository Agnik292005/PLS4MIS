"""
cmf_val.py

Sliding-window inference for DoDNet on the CMF data. Additive: the stock
utils/val3D.py is not touched.

Differences from the stock version:
  * scans shorter than the patch on any axis are padded with 0 (the scan mean
    after z-score) and the prediction is cropped back to the original shape,
  * tiles may overlap (stride < patch); overlapping probabilities are averaged,
  * one forward pass per class (batch of 1) to keep memory low.
Array order is (z, y, x); patch_size is given in (z, y, x) here.
"""
import numpy as np
import torch


def _starts(dim, patch, stride):
    if dim <= patch:
        return [0]
    s = list(range(0, dim - patch + 1, stride))
    if s[-1] != dim - patch:
        s.append(dim - patch)
    return s


@torch.no_grad()
def predict_volume(model, image, num_classes, patch_zyx, overlap=0.0, device="cuda"):
    """image: float tensor/array (z, y, x), already z-scored.
    Returns an int label array (z, y, x) of the ORIGINAL shape."""
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
    nfg = num_classes - 1
    prob = torch.zeros((nfg, Z, Y, X), dtype=torch.float32)
    cnt = torch.zeros((Z, Y, X), dtype=torch.float32)
    model.eval()
    for zs in _starts(Z, patch_zyx[0], strides[0]):
        for ys in _starts(Y, patch_zyx[1], strides[1]):
            for xs in _starts(X, patch_zyx[2], strides[2]):
                sl = (slice(zs, zs + patch_zyx[0]), slice(ys, ys + patch_zyx[1]), slice(xs, xs + patch_zyx[2]))
                tile = image[sl].to(device)[None, None]
                for c in range(nfg):
                    enc = torch.zeros((1, nfg), device=device)
                    enc[0, c] = 1.0
                    out = model(tile, enc)
                    prob[(c,) + sl] += out[0, 0].float().sigmoid().cpu()
                cnt[sl] += 1.0
    prob /= cnt.clamp(min=1.0)
    full = torch.cat([torch.full((1, Z, Y, X), 0.5), prob], 0)
    lab = full.argmax(0).numpy().astype(np.int16)
    sl = tuple(slice(b, b + o) for (b, _), o in zip(pads, orig))
    return lab[sl]
