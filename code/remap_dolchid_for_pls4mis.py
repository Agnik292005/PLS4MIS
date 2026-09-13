"""
remap_dolchid_for_pls4mis.py

Remaps DOLCHID's binary lesion labels (0=background, 1=lesion) into the
combined label scheme used across the whole CMF dataset for PLS4MIS:

    0 = background
    1 = mandible
    2 = maxilla
    3 = IAN
    4 = mandible-teeth
    5 = maxilla-teeth
    6 = lesion   (DOLCHID's old value 1, remapped)

Confirmed with Sukhraj directly (Sep 12): lesion gets the next available
label number after anatomy.

Also copies images and remapped labels into the imagesTr/labelsTr folder
structure PLS4MIS's AbdomenOrgan dataloader expects.

Read-only on the original DOLCHID files -- writes new remapped copies,
never modifies the originals.

Usage:
    python3 remap_dolchid_for_pls4mis.py
"""

import os
import shutil
import nibabel as nib
import numpy as np

DOLCHID_IMG_DIR = "/DATA/scratch/sukhraj/nnUNet_raw/Dataset502_DOLCHID/imagesTr"
DOLCHID_LBL_DIR = "/DATA/scratch/sukhraj/nnUNet_raw/Dataset502_DOLCHID/labelsTr"

OUTPUT_ROOT = "/DATA/scratch/sukhraj/PLS4MIS_CMF_data"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "imagesTr")
OUTPUT_LBL_DIR = os.path.join(OUTPUT_ROOT, "labelsTr")

LESION_OLD_VALUE = 1
LESION_NEW_VALUE = 6


def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    label_files = sorted(f for f in os.listdir(DOLCHID_LBL_DIR) if f.endswith(".nii.gz"))
    print(f"Found {len(label_files)} DOLCHID label files to remap.")

    remapped = 0
    errors = []

    for label_fname in label_files:
        case_id = label_fname[: -len(".nii.gz")]
        image_fname = f"{case_id}_0000.nii.gz"

        src_label_path = os.path.join(DOLCHID_LBL_DIR, label_fname)
        src_image_path = os.path.join(DOLCHID_IMG_DIR, image_fname)

        if not os.path.exists(src_image_path):
            errors.append(f"MISSING IMAGE for {case_id}: {src_image_path}")
            continue

        try:
            lbl_nii = nib.load(src_label_path)
            lbl_data = lbl_nii.get_fdata()

            unique_vals = set(np.unique(lbl_data).tolist())
            unexpected = unique_vals - {0.0, float(LESION_OLD_VALUE)}
            if unexpected:
                errors.append(f"UNEXPECTED VALUES in {case_id}: {unexpected}, skipping remap for safety")
                continue

            remapped_data = np.where(lbl_data == LESION_OLD_VALUE, LESION_NEW_VALUE, lbl_data).astype(lbl_data.dtype)

            remapped_nii = nib.Nifti1Image(remapped_data, lbl_nii.affine, lbl_nii.header)
            out_label_path = os.path.join(OUTPUT_LBL_DIR, label_fname)
            nib.save(remapped_nii, out_label_path)

            out_image_path = os.path.join(OUTPUT_IMG_DIR, image_fname)
            if not os.path.exists(out_image_path):
                shutil.copy2(src_image_path, out_image_path)

            remapped += 1
        except Exception as e:
            errors.append(f"ERROR on {case_id}: {e}")

    print(f"\nRemapped {remapped}/{len(label_files)} cases successfully.")
    print(f"Output: {OUTPUT_ROOT}")

    if errors:
        print(f"\n{len(errors)} issues:")
        for e in errors:
            print(f"  {e}")
    else:
        print("No issues.")


if __name__ == "__main__":
    main()
