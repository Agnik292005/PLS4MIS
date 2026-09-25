# CMF anatomy-and-lesion benchmark: colleague handoff plan

All code paths below are relative to the **PLS4MIS repository root**.

## 1. Objective and settings

Compare **DoDNet, U-Net, MELoss, LeafDice, MultiTalent, and PL-Seg** on the **combined anatomy-plus-lesion dataset**. No separate anatomy-only benchmark is required.

Complete DoDNet first, then adapt the remaining methods one at a time.

| Setting | Value |
|---|---|
| Dataset | Existing anatomy data plus the prepared mandible-only DOLCHID subset |
| Classes | **7:** background, mandible, maxilla, IAN, mandibular teeth, maxillary teeth, lesion |
| Spacing | Existing **0.3 mm isotropic** |
| Patch | **192 × 192 × 192** |
| Initial batch size | **1**, with mixed precision |
| Training | **500 epochs** |
| Validation | Every **2 epochs** |
| Splits | Preserve the locked train/validation assignments |

The combined preparation entrypoint is [code/data_prep/combine_anatomy_and_dolchid.py](code/data_prep/combine_anatomy_and_dolchid.py#L27). Lesion must retain label **6**.

## 2. Tasks in priority order

| Order | Priority | Task |
|---|---|---|
| 1 | **Critical** | Verify image/label pairing, split integrity, lesion remapping, and which classes are annotated in each case. |
| 2 | **High** | Implement whole-volume z-score preprocessing and save a reusable cache. |
| 3 | **High** | Use one case row and one randomly selected eligible task per DoDNet case visit. |
| 4 | **Critical** | Implement annotation-aware evaluation and verify checkpoint loading and resume. |
| 5 | **High** | Qualify complete seven-class DoDNet training and validation on the 24 GB GPU. |
| 6 | **High** | Adapt U-Net → MELoss → LeafDice → MultiTalent → PL-Seg, verifying each method’s loss and label handling. |
| 7 | **Later** | Benchmark Blosc2 against NumPy before adopting it. |

## 3. Data preparation and sampling

Keep the nnU-Net naming convention:

```text
Image: imagesTr/case_0000.nii.gz
Label: labelsTr/case.nii.gz
```

Correct the lookup in [code/dataloader/CMFDataset.py:63](code/dataloader/CMFDataset.py#L63) to remove the image-channel suffix when locating labels.

Normalize each whole image before cropping:

\[
I_{\text{normalized}}=\frac{I-\mu_{\text{volume}}}{\sigma_{\text{volume}}}
\]

Use the same procedure during training and inference. Save normalized float32 images, integer labels, statistics, and geometry. Start with memory-mapped NumPy storage; remove subsequent min–max normalization from the new pipeline.

For DoDNet:

- Use **one dataset row per case**.
- Select one task uniformly from foreground classes present in the full-volume label, verified against annotation metadata.
- Retain the random selection approach in [code/train_DoDNet.py:175](code/train_DoDNet.py#L175).
- A lesion-only DOLCHID case selects lesion; an anatomy case selects one of its available anatomical structures.
- Keep random cropping as the initial augmentation and record positive-patch coverage.

Random task selection limits computation while distributing supervision across repeated case visits. It does not guarantee equal global exposure across classes.

## 4. Training and evaluation

Run **500 epochs**, with one visit to every training case per epoch and no dropped final batch.

For a case with five eligible anatomy tasks, each task receives approximately **100 selections** over 500 epochs. A lesion-only case receives approximately **500 lesion selections**. Log actual selections and positive crops; these counts do not guarantee convergence.

Validate every **2 epochs** and report:

- Dice for each anatomical structure.
- Mean anatomy Dice.
- Lesion Dice.
- Mean Dice across all six foreground classes.

**Evaluate a class only where it is annotated.** Do not treat missing lesion annotations in anatomy cases—or missing anatomy annotations in DOLCHID cases—as confirmed negatives.

Select the best checkpoint using the mean of the six eligible per-class foreground Dice scores, excluding background. Use the same evaluation rules across methods.

Inspect learning curves at epochs 100, 200, and 500. Any 1,000-epoch comparison should use that budget for all methods.

## 5. Completion checks and reporting

Before proceeding beyond DoDNet, confirm:

- Correct data pairing, normalization, and annotation-aware evaluation.
- Reproducible case, patch, and task sampling.
- Successful checkpoint reload and resume.
- Full training and validation fit the actual GPU with memory headroom.
- Small-data tests demonstrate learning for both anatomy and lesion examples.

Apply equivalent checks to each subsequent method. Keep U-Net identified as the naïve partial-label baseline and verify the specialized loss implementations before accepting results.

Use seed **42** for qualification and **42, 123, and 2026** for the final comparison. Report anatomy and lesion performance separately, alongside variability, optimizer updates, runtime, and peak GPU memory.
