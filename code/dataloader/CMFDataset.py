"""
CMFDataset.py

DoDNet-style dataset for PLS4MIS, addressing the coverage issue found in
the stock AbdomenOrgan + train_DoDNet.py combination: normally, one case
contributes exactly ONE sample per epoch, and the training loop randomly
picks ONE present structure from that case to actually train on
(train_DoDNet.py's `rand_cls = fg[random.randint(0, fg.shape[0]-1)]`). A
case with several labeled structures does not get all of them trained
every epoch, just one, chosen randomly, so rarer structures can be
under-covered relative to common ones unless many epochs pass for the
randomness to average out.

This class instead expands the case list ONE TIME at load, so every
case appears once PER PRESENT STRUCTURE, exactly mirroring the approach
already proven for DoDNet's own dental_train.txt (build_dodnet_dental_
lists.py): a case with 5 present structures contributes 5 rows, so every
structure gets trained every epoch, deterministic coverage instead of
random.

Each row carries an explicit `forced_class` (1-indexed, matching the
label's real class values) alongside the usual image/label pair. The
corresponding one-line change needed in train_DoDNet.py's training loop
is to use sample['forced_class'] directly instead of randomly sampling
from the one-hot cur_task vector, when forced_class is present.

Does not change the label CONTENT or normalization in any way, only which
(and how many) samples are generated per case, purely additive relative
to the original AbdomenOrgan class, which is left completely untouched.
"""

import os
from glob import glob
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset


class CMFAnatomyDataset(Dataset):
    def __init__(self, nii_dir, mode='train', transform=None):
        """
        :param nii_dir: Data storage location (same imagesTr/labelsTr/
            imagesVal/labelsVal layout as AbdomenOrgan).
        :param mode: 'train' or 'val'. Row-expansion only applies to
            'train' (validation should still see one full case at a time,
            matching how PLS4MIS's own validate_slice() works).
        """
        self.nii_dir = nii_dir
        self.mode = mode
        self.transform = transform
        self.sample_list = []

        if self.mode == 'train':
            image_dir = os.path.join(self.nii_dir, 'imagesTr/')
            print('==> Loading {} data from: {}'.format(mode, image_dir))
            image_list = sorted(glob(image_dir + '*.nii.gz'))

            total_cases = 0
            total_rows = 0
            skipped_no_foreground = []

            for image_path in image_list:
                gt_path = image_path.replace('imagesTr', 'labelsTr')  # same filename, matches AbdomenOrgan's convention and the actual files on disk
                if not os.path.exists(gt_path):
                    continue

                label_img = sitk.ReadImage(gt_path)
                label_arr = sitk.GetArrayFromImage(label_img)
                present_classes = np.unique(label_arr)
                present_classes = present_classes[present_classes != 0]

                total_cases += 1
                if present_classes.size == 0:
                    skipped_no_foreground.append(os.path.basename(image_path))
                    continue

                for cls in present_classes:
                    self.sample_list.append({
                        'image': image_path,
                        'label': gt_path,
                        'forced_class': int(cls),
                    })
                    total_rows += 1

            print(f"CMFAnatomyDataset: {total_cases} cases -> {total_rows} rows "
                  f"(one row per present structure per case)")
            if skipped_no_foreground:
                print(f"  {len(skipped_no_foreground)} cases had no foreground at all, skipped: "
                      f"{skipped_no_foreground[:5]}{'...' if len(skipped_no_foreground) > 5 else ''}")

        elif self.mode == 'val':
            image_dir = os.path.join(self.nii_dir, 'imagesVal/')
            print('==> Loading {} data from: {}'.format(mode, image_dir))
            image_list = sorted(glob(image_dir + '*.nii.gz'))
            for image_path in image_list:
                gt_path = image_path.replace('imagesVal', 'labelsVal')  # same filename, matches AbdomenOrgan's convention and the actual files on disk
                self.sample_list.append({'image': image_path, 'label': gt_path})
            print(f"CMFAnatomyDataset: {len(self.sample_list)} val cases (one row per case, "
                  f"unchanged from AbdomenOrgan, matches how validate_slice() expects full cases)")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        entry = self.sample_list[index]
        _dataimg = sitk.ReadImage(entry['image'])
        _image = sitk.GetArrayFromImage(_dataimg)
        _datagt = sitk.ReadImage(entry['label'])
        _target = sitk.GetArrayFromImage(_datagt)
        _img_name = os.path.basename(entry['image'])

        sample = {'image': _image, 'label': _target, 'img_name': _img_name}
        if 'forced_class' in entry:
            sample['forced_class'] = entry['forced_class']

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
