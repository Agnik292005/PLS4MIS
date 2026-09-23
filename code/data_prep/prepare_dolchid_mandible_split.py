"""
prepare_dolchid_mandible_split.py (v2, corrected)

Rebuilds the DOLCHID data prep for PLS4MIS per Sukhraj's Sep 20 mail
(mandible-only, all 4 lesion subtypes represented in train and val), but
CORRECTLY derives the split from the existing LOCKED fold-0 split
(splits_DOLCHID_locked.json, from the dental-cbct-segmentation repo) that
DoDNet, nnU-Net, and MultiTalent all already use, rather than generating a
brand new, unrelated random split.

v1 of this script generated an independent 149/37 split, which would have
made PLS4MIS's future DOLCHID lesion results NOT directly comparable to
DoDNet's existing DOLCHID lesion results (n=53, fold 0's val set) -- a real
problem caught before it was used for anything, not after.

This version instead:
  1. Loads fold 0 of splits_DOLCHID_locked.json (209 train / 53 val, all
     262 cases, mandible+maxilla mixed, exactly what DoDNet trained on).
  2. Filters BOTH train and val down to mandible-only cases, preserving
     the original train/val assignment for every case that survives the
     filter (never moves a case between train and val).
  3. Reports the resulting subtype balance in each split. If any subtype
     ends up completely missing from val, flags it loudly rather than
     silently proceeding.

Usage:
    python3 prepare_dolchid_mandible_split.py
"""

import os
import csv
import json
import shutil
import openpyxl
import nibabel as nib
import numpy as np
from collections import Counter

CASE_MAPPING_CSV = "/DATA/scratch/sukhraj/nnUNet_raw/Dataset502_DOLCHID/case_mapping_log.csv"
LESION_REVIEW_XLSX = "/DATA/scratch/sukhraj/dolchid_lesion_review.xlsx"
LOCKED_SPLITS_JSON = "/DATA/scratch/sukhraj/splits_DOLCHID_locked.json"
FOLD = 0

DOLCHID_IMG_DIR = "/DATA/scratch/sukhraj/nnUNet_raw/Dataset502_DOLCHID/imagesTr"
DOLCHID_LBL_DIR = "/DATA/scratch/sukhraj/nnUNet_raw/Dataset502_DOLCHID/labelsTr"

OUTPUT_ROOT = "/DATA/scratch/sukhraj/PLS4MIS_CMF_data_dolchid_mandible"
TRAIN_IMG_DIR = os.path.join(OUTPUT_ROOT, "imagesTr")
TRAIN_LBL_DIR = os.path.join(OUTPUT_ROOT, "labelsTr")
VAL_IMG_DIR = os.path.join(OUTPUT_ROOT, "imagesVal")
VAL_LBL_DIR = os.path.join(OUTPUT_ROOT, "labelsVal")

LESION_OLD_VALUE = 1
LESION_NEW_VALUE = 6


def load_case_mapping():
    mapping = {}
    with open(CASE_MAPPING_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["nnUNet_ID"]] = (row["Original_ID"], row["Lesion_Type"])
    return mapping


def load_jaw_lookup():
    wb = openpyxl.load_workbook(LESION_REVIEW_XLSX, data_only=True)
    ws = wb["case_review - Copy"]
    jaw_lookup = {}
    subtype_lookup = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        case_id, subtype, _, _, jaw = row
        jaw_lookup[case_id] = jaw
        subtype_lookup[case_id] = subtype
    return jaw_lookup, subtype_lookup


def main():
    case_mapping = load_case_mapping()
    jaw_lookup, subtype_lookup = load_jaw_lookup()

    with open(LOCKED_SPLITS_JSON) as f:
        folds = json.load(f)
    fold_data = folds[FOLD]
    locked_train = fold_data["train"]
    locked_val = fold_data["val"]
    print(f"Locked fold {FOLD}: {len(locked_train)} train, {len(locked_val)} val "
          f"({len(locked_train) + len(locked_val)} total, matches DoDNet's own DOLCHID split)")

    unmatched = []
    subtype_mismatches = []

    def classify(nnunet_id):
        if nnunet_id not in case_mapping:
            unmatched.append((nnunet_id, "NOT IN case_mapping_log.csv"))
            return None
        original_id, subtype_from_mapping = case_mapping[nnunet_id]
        if original_id not in jaw_lookup:
            unmatched.append((nnunet_id, f"original {original_id} NOT IN review sheet"))
            return None
        subtype_from_sheet = subtype_lookup[original_id]
        if subtype_from_sheet != subtype_from_mapping:
            subtype_mismatches.append((nnunet_id, original_id, subtype_from_mapping, subtype_from_sheet))
            return None
        return jaw_lookup[original_id], subtype_from_mapping

    train_ids = []
    train_subtypes = []
    for nnunet_id in locked_train:
        result = classify(nnunet_id)
        if result and result[0] == "mandible":
            train_ids.append(nnunet_id)
            train_subtypes.append(result[1])

    val_ids = []
    val_subtypes = []
    for nnunet_id in locked_val:
        result = classify(nnunet_id)
        if result and result[0] == "mandible":
            val_ids.append(nnunet_id)
            val_subtypes.append(result[1])

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} cases could not be classified:")
        for nid, reason in unmatched[:10]:
            print(f"  {nid}: {reason}")
    if subtype_mismatches:
        print(f"\nWARNING: {len(subtype_mismatches)} subtype mismatches between sources:")
        for nid, oid, sub_map, sub_sheet in subtype_mismatches[:10]:
            print(f"  {nid} ({oid}): mapping says {sub_map}, sheet says {sub_sheet}")
    if unmatched or subtype_mismatches:
        print("\nABORTING before touching any files: resolve the warnings above first.")
        return

    print(f"\nMandible-only, derived from locked fold {FOLD}: {len(train_ids)} train, {len(val_ids)} val")

    train_counts = Counter(train_subtypes)
    val_counts = Counter(val_subtypes)
    print("Train subtype counts:", dict(train_counts))
    print("Val subtype counts:  ", dict(val_counts))

    all_subtypes = set(train_counts) | set(val_counts)
    missing_from_val = [s for s in all_subtypes if val_counts.get(s, 0) == 0]
    if missing_from_val:
        print(f"\nWARNING: subtype(s) {missing_from_val} have ZERO cases in val after filtering "
              f"the locked split to mandible-only. Flagging for Sukhraj rather than silently "
              f"proceeding or silently rebalancing (which would break comparability with "
              f"DoDNet's existing split).")

    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LBL_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)

    def process_split(ids, img_out_dir, lbl_out_dir):
        errors = []
        processed = 0
        for nnunet_id in ids:
            src_img = os.path.join(DOLCHID_IMG_DIR, f"{nnunet_id}_0000.nii.gz")
            src_lbl = os.path.join(DOLCHID_LBL_DIR, f"{nnunet_id}.nii.gz")
            if not os.path.exists(src_img) or not os.path.exists(src_lbl):
                errors.append(f"MISSING FILE for {nnunet_id}")
                continue

            lbl_nii = nib.load(src_lbl)
            lbl_data = lbl_nii.get_fdata()
            unique_vals = set(np.unique(lbl_data).tolist())
            unexpected = unique_vals - {0.0, float(LESION_OLD_VALUE)}
            if unexpected:
                errors.append(f"UNEXPECTED VALUES in {nnunet_id}: {unexpected}, skipped")
                continue

            remapped = np.where(lbl_data == LESION_OLD_VALUE, LESION_NEW_VALUE, lbl_data).astype(lbl_data.dtype)
            remapped_nii = nib.Nifti1Image(remapped, lbl_nii.affine, lbl_nii.header)
            nib.save(remapped_nii, os.path.join(lbl_out_dir, f"{nnunet_id}.nii.gz"))
            shutil.copy2(src_img, os.path.join(img_out_dir, f"{nnunet_id}_0000.nii.gz"))
            processed += 1
        return processed, errors

    train_ok, train_errors = process_split(train_ids, TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    val_ok, val_errors = process_split(val_ids, VAL_IMG_DIR, VAL_LBL_DIR)

    print(f"\nTrain: {train_ok}/{len(train_ids)} processed successfully")
    print(f"Val: {val_ok}/{len(val_ids)} processed successfully")
    for e in train_errors + val_errors:
        print(f"  {e}")

    print(f"\nOutput: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
