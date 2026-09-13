"""
setup_anatomy_only_for_pls4mis.py

Organizes the existing anatomy data (DentVoxel + ToothFairy3, already using
the 1-5 label scheme: mandible, maxilla, IAN, mandible-teeth, maxilla-teeth)
into the imagesTr/labelsTr structure PLS4MIS's AbdomenOrgan dataloader
expects, for the anatomy-only stage (stage 1 of the plan agreed with
Sukhraj: anatomy only -> anatomy+DOLCHID -> anatomy+DOLCHID+AIIMS).

No label remapping needed here -- anatomy already uses the exact 1-5
scheme this whole combined numbering is built around (confirmed directly
from TF3_TASK_LABEL_VALUE in MOTSDataset_dental.py). This script only
copies files into the expected folder layout, split into a train and val
set using the SAME fold-0 splits_final.json already used for every other
model in this project, so results stay comparable.

num_classes=6 for this stage (5 anatomy classes + background), since
DOLCHID/lesion isn't included yet.

Usage:
    python3 setup_anatomy_only_for_pls4mis.py
"""

import os
import json
import shutil

DENTVOXEL_IMG_DIR = "/DATA/scratch/sukhraj/anatomy_training_data/DentVoxel_Anatomy_LPS/DentVoxel_Anatomy_LPS/images-Tr"
DENTVOXEL_LBL_DIR = "/DATA/scratch/sukhraj/anatomy_training_data/DentVoxel_Anatomy_LPS/DentVoxel_Anatomy_LPS/labels-Tr"
TOOTHFAIRY3_IMG_DIR = "/DATA/scratch/sukhraj/anatomy_training_data/ToothFairy3_Anatomy_LPS/ToothFairy3_Anatomy_LPS/imagesTr"
TOOTHFAIRY3_LBL_DIR = "/DATA/scratch/sukhraj/anatomy_training_data/ToothFairy3_Anatomy_LPS/ToothFairy3_Anatomy_LPS/labelsTr"

SPLITS_PATH = "/home/sukhraj/splits_final.json"
FOLD = 0

OUTPUT_ROOT = "/DATA/scratch/sukhraj/PLS4MIS_CMF_data_anatomy_only"
TRAIN_IMG_DIR = os.path.join(OUTPUT_ROOT, "imagesTr")
TRAIN_LBL_DIR = os.path.join(OUTPUT_ROOT, "labelsTr")
VAL_IMG_DIR = os.path.join(OUTPUT_ROOT, "imagesVal")
VAL_LBL_DIR = os.path.join(OUTPUT_ROOT, "labelsVal")


def resolve_paths(case_id):
    if case_id.startswith("DentVoxel_img"):
        num_part = case_id[len("DentVoxel_img"):]
        img_path = os.path.join(DENTVOXEL_IMG_DIR, f"DentVoxel_img{num_part}_0000.nii.gz")
        lbl_path = os.path.join(DENTVOXEL_LBL_DIR, f"DentVoxel_label{num_part}.nii.gz")
        return img_path, lbl_path
    if case_id.startswith("ToothFairy3"):
        img_path = os.path.join(TOOTHFAIRY3_IMG_DIR, f"{case_id}_0000.nii.gz")
        lbl_path = os.path.join(TOOTHFAIRY3_LBL_DIR, f"{case_id}.nii.gz")
        return img_path, lbl_path
    return None, None


def copy_split(case_ids, img_out_dir, lbl_out_dir, log_lines):
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    copied = 0
    for case_id in case_ids:
        img_path, lbl_path = resolve_paths(case_id)
        if img_path is None:
            log_lines.append(f"UNRECOGNIZED CASE ID: {case_id}")
            continue
        if not os.path.exists(img_path):
            log_lines.append(f"MISSING IMAGE: {case_id} -> {img_path}")
            continue
        if not os.path.exists(lbl_path):
            log_lines.append(f"MISSING LABEL: {case_id} -> {lbl_path}")
            continue

        img_fname = f"{case_id}_0000.nii.gz"
        lbl_fname = f"{case_id}.nii.gz"

        out_img = os.path.join(img_out_dir, img_fname)
        out_lbl = os.path.join(lbl_out_dir, lbl_fname)

        if not os.path.exists(out_img):
            shutil.copy2(img_path, out_img)
        if not os.path.exists(out_lbl):
            shutil.copy2(lbl_path, out_lbl)
        copied += 1
    return copied


def main():
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    train_ids = splits[FOLD]["train"]
    val_ids = splits[FOLD]["val"]

    print(f"Fold {FOLD}: {len(train_ids)} train, {len(val_ids)} val cases (same split used for DoDNet/nnU-Net/MultiTalent)")

    log_lines = []
    train_copied = copy_split(train_ids, TRAIN_IMG_DIR, TRAIN_LBL_DIR, log_lines)
    val_copied = copy_split(val_ids, VAL_IMG_DIR, VAL_LBL_DIR, log_lines)

    print(f"\nCopied {train_copied}/{len(train_ids)} train cases to {TRAIN_IMG_DIR}")
    print(f"Copied {val_copied}/{len(val_ids)} val cases to {VAL_IMG_DIR}")
    print(f"\nnum_classes for this stage: 6 (5 anatomy structures + background)")

    if log_lines:
        print(f"\n{len(log_lines)} issues:")
        for line in log_lines:
            print(f"  {line}")
    else:
        print("\nNo issues, every case resolved and copied cleanly.")


if __name__ == "__main__":
    main()
