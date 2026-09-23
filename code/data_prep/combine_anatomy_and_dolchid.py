"""
combine_anatomy_and_dolchid.py

Builds the Stage 2 (anatomy + DOLCHID) data folder for PLS4MIS, by copying
both the anatomy-only folder (PLS4MIS_CMF_data_anatomy_only/, built by
setup_anatomy_only_for_pls4mis.py) and the DOLCHID-mandible folder
(PLS4MIS_CMF_data_dolchid_mandible/, built by
prepare_dolchid_mandible_split.py) into one shared destination.

Needed because AbdomenOrgan (and our own dataset classes) only read from
ONE imagesTr/labelsTr folder at a time -- they can't combine two separate
folders on their own, so this script does that combining once, ahead of
time, rather than at training time.

Read-only on both source folders. Reports any filename collision found
between the two sources (shouldn't happen, since DentVoxel/TF3 and
DOLCHID use entirely different naming conventions, but checked explicitly
rather than assumed).

Usage:
    python3 combine_anatomy_and_dolchid.py
"""

import os
import shutil

ANATOMY_ROOT = "/DATA/scratch/sukhraj/PLS4MIS_CMF_data_anatomy_only"
DOLCHID_ROOT = "/DATA/scratch/sukhraj/PLS4MIS_CMF_data_dolchid_mandible"
OUTPUT_ROOT = "/DATA/scratch/sukhraj/PLS4MIS_CMF_data_anatomy_plus_dolchid"


def combine_folder(sub, sources, destination):
    os.makedirs(destination, exist_ok=True)
    collisions = []
    copied = 0
    for source_root, label in sources:
        source_dir = os.path.join(source_root, sub)
        if not os.path.isdir(source_dir):
            print(f"  SKIP (not found): {source_dir}")
            continue
        for fname in os.listdir(source_dir):
            dest_path = os.path.join(destination, fname)
            if os.path.exists(dest_path):
                collisions.append((fname, label))
                continue
            shutil.copy2(os.path.join(source_dir, fname), dest_path)
            copied += 1
    return copied, collisions


def main():
    sources = [(ANATOMY_ROOT, "anatomy"), (DOLCHID_ROOT, "dolchid")]

    for sub in ["imagesTr", "labelsTr", "imagesVal", "labelsVal"]:
        destination = os.path.join(OUTPUT_ROOT, sub)
        copied, collisions = combine_folder(sub, sources, destination)
        print(f"{sub}: {copied} files copied")
        if collisions:
            print(f"  WARNING: {len(collisions)} filename collisions, not overwritten:")
            for fname, label in collisions[:10]:
                print(f"    {fname} (from {label})")

    print(f"\nOutput: {OUTPUT_ROOT}")

    for sub in ["imagesTr", "labelsTr", "imagesVal", "labelsVal"]:
        count = len(os.listdir(os.path.join(OUTPUT_ROOT, sub)))
        print(f"  {sub}: {count} total files")


if __name__ == "__main__":
    main()
