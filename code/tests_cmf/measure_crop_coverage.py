"""Read-only check: how often does a random PATCH^3 crop contain each class?
No GPU, nothing is modified. Uses 3D block counts + integral image (fast).
"""
import argparse, csv, glob, os
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor

CLASS_NAMES = {1: "mandible", 2: "maxilla", 3: "IAN", 4: "mandibular_teeth", 5: "maxillary_teeth"}


def block_counts(mask, b):
    s = [int(np.ceil(n / b)) * b for n in mask.shape]
    pad = np.zeros(s, dtype=np.uint16)
    pad[:mask.shape[0], :mask.shape[1], :mask.shape[2]] = mask
    return pad.reshape(s[0]//b, b, s[1]//b, b, s[2]//b, b).sum(axis=(1, 3, 5), dtype=np.int64)


def integral(c):
    I = np.zeros([n + 1 for n in c.shape], dtype=np.int64)
    I[1:, 1:, 1:] = c.cumsum(0).cumsum(1).cumsum(2)
    return I


def box(I, x, y, z, w):
    X, Y, Z = x + w[0], y + w[1], z + w[2]
    return (I[X, Y, Z] - I[x, Y, Z] - I[X, y, Z] - I[X, Y, z]
            + I[x, y, Z] + I[x, Y, z] + I[X, y, z] - I[x, y, z])


def process(args):
    path, patch, draws, b, min_vox, seed = args
    lab = np.asanyarray(nib.load(path).dataobj).astype(np.int16)
    rng = np.random.default_rng(seed)
    shape = lab.shape
    short = [i for i, n in enumerate(shape) if n < patch]
    pb = [min(patch, n) // b for n in shape]
    nb = [int(np.ceil(n / b)) for n in shape]
    starts = [rng.integers(0, max(nb[i] - pb[i], 0) + 1, size=draws) for i in range(3)]
    out = {"case": os.path.basename(path), "short_axes": ",".join(map(str, short))}
    for c, name in CLASS_NAMES.items():
        m = lab == c
        tot = int(m.sum())
        out[f"{name}_voxels"] = tot
        if tot == 0:
            out[f"{name}_p_any"] = out[f"{name}_p_min"] = out[f"{name}_frac"] = float("nan")
            continue
        I = integral(block_counts(m, b))
        cnt = box(I, starts[0], starts[1], starts[2], pb)
        out[f"{name}_p_any"] = float((cnt > 0).mean())
        out[f"{name}_p_min"] = float((cnt >= min_vox).mean())
        out[f"{name}_frac"] = float((cnt / tot).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--min_voxels", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.label_dir, "*.nii.gz")))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit("no .nii.gz labels in " + a.label_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(f, a.patch, a.draws, a.block, a.min_voxels, a.seed + i) for i, f in enumerate(files)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(process, jobs))
    with open(os.path.join(a.out_dir, "crop_coverage_per_case.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} cases, patch {a.patch}, min_voxels {a.min_voxels}")
    print(f"{'class':18s}{'cases_present':>14s}{'p_any':>8s}{'p_>=min':>9s}{'mean_frac':>10s}")
    for name in CLASS_NAMES.values():
        pres = [r for r in rows if r[f"{name}_voxels"] > 0]
        if not pres:
            print(f"{name:18s}{0:14d}"); continue
        print(f"{name:18s}{len(pres):14d}{np.mean([r[name+'_p_any'] for r in pres]):8.3f}"
              f"{np.mean([r[name+'_p_min'] for r in pres]):9.3f}{np.mean([r[name+'_frac'] for r in pres]):10.3f}")
    short = [r["case"] for r in rows if r["short_axes"]]
    print(f"volumes with an axis shorter than patch: {len(short)}")
    for s in short[:20]:
        print("  ", s)


if __name__ == "__main__":
    main()
