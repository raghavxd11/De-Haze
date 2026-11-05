# Benchmark

## Purpose

A fixed, repeatable way to measure how well each dehazing method restores the
ground-truth image on this project's dataset. All methods are scored on the same
test images, at the same size, with the same metric code (`evaluate.py`).

## Methods

| Method | Where it runs | Notes |
| --- | --- | --- |
| Ours: U-Net | `train.py --model unet` | 31,379,075 parameters, sigmoid output, MSE loss |
| Ours: Pix2Pix | `train.py --model pix2pix` | Generator is the same U-Net (31,379,075 parameters); 70x70 PatchGAN discriminator used only in training; GAN + 100 x L1 loss |
| Dark Channel Prior (DCP) | `baselines/dcp.py` | He et al., CVPR 2009. No training. 15x15 patch, omega 0.95, t0 0.1, guided filter r=24, eps=1e-3 (fixed, not tuned on any split) |
| AOD-Net | external | Li et al., ICCV 2017. Run outside this repo; outputs scored with `--external-dir` |
| FFA-Net | external | Qin et al., AAAI 2020. Run outside this repo; outputs scored with `--external-dir` |

Our two models see the hazy image after CLAHE/HLS preprocessing (`preprocess.py`).
DCP and the external models get the raw hazy image, only resized to 256x256.

## Dataset

Checked by reading every file:

- 2,036 hazy / ground-truth pairs (`dataset/Hazy_Full`, `dataset/GT_Full`).
- IDs 01-55 (55 pairs) are outdoor images, real PNG files, 1600x1200.
- The other 1,981 pairs (about 97%) are indoor. They are JPEG data saved with a `.png`
  extension, in five sizes: 2708x1800 (1,642), 2209x1800 (107), 1806x1800 (82),
  2708x1700 (82), 1809x1800 (68).
- Where the images come from is not documented in this repository.
- For training and evaluation every image is stretched to 256x256 with `cv2.INTER_AREA`
  (the aspect ratio is not kept).

## Split

The split is stored in `splits/split_v1.json` and is never recomputed during training
or evaluation.

Many indoor images show the same room. A random split would put near-copies of a test
image into the training set, so similar ground-truth images are grouped first and each
group goes entirely into one split:

- Features: MobileNetV2 (ImageNet weights), cosine distance.
- Clustering: agglomerative, average linkage, distance threshold 0.20, giving 113 groups.
- Groups larger than 30 images always go to train. The rest are shuffled with seed 42 and
  assigned to val and test until each has about 10% of the images, separately for
  indoor and outdoor.
- A threshold of 0.15 was tried first. It left 58.1% of val+test images within cosine
  distance 0.14 of some training image, which was too leaky. At 0.20 this is 10.0% for
  val+test and 4.2% for test. For comparison, a random split of the same size gives 97.9%.
- Grouping is approximate: some repeated rooms can still appear in two splits.

| Split | Images | Indoor | Outdoor | Scene groups |
| --- | --- | --- | --- | --- |
| train | 1,608 | 1,565 | 43 | 74 |
| val | 215 | 209 | 6 | 20 |
| test | 213 | 207 | 6 | 19 |

The val split is used to watch training. Final numbers are reported on test only.

## Metrics protocol

- Each method produces a 256x256 uint8 BGR image per test image.
- The reference is the ground truth stretched to 256x256 with `INTER_AREA`.
- PSNR and SSIM from `tf.image` with `max_val=1.0` (images divided by 255).
  Infinite PSNR (identical images) is capped at 100 dB.
- Reported for overall, indoor and outdoor:
  - number of images and scene groups,
  - mean over images,
  - mean of per-scene-group means (so large groups do not dominate),
  - 95% bootstrap confidence interval of the group mean (2,000 resamples of scene
    groups, seed 42).
- Parameter count, and average ms per image at batch size 1 after 3 warm-up runs,
  with the device printed.
- Runs with `--limit` are marked as not final.

## Reproduce

```
# train our models (outputs/models/<model>/)
python train.py --model unet
python train.py --model pix2pix

# score our models and DCP on the test split
python evaluate.py --model-path outputs/models/unet/unet.keras \
                   --model-path outputs/models/pix2pix/p2p.keras --compare-dcp

# external models: export the raw 256x256 test inputs, run the model elsewhere,
# save its outputs as <hazy stem>.png, then score them
python evaluate.py --export-test-inputs exported/test_inputs
python evaluate.py --external-dir results/aodnet --external-name AOD-Net
python evaluate.py --external-dir results/ffanet --external-name FFA-Net
```

Results are written to `outputs/results.md`, `outputs/results.csv` and
`outputs/results_per_image.csv`. Each `evaluate.py` run overwrites these files.

## Results (test split)

Not measured yet.

| Method | Subset | Images | Groups | PSNR (group mean) [95% CI] | SSIM (group mean) [95% CI] | Params | ms/image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | overall | 213 | 19 | TBD | TBD | TBD | TBD |
| U-Net | indoor | 207 | 13 | TBD | TBD | TBD | TBD |
| U-Net | outdoor | 6 | 6 | TBD | TBD | TBD | TBD |
| Pix2Pix | overall | 213 | 19 | TBD | TBD | TBD | TBD |
| Pix2Pix | indoor | 207 | 13 | TBD | TBD | TBD | TBD |
| Pix2Pix | outdoor | 6 | 6 | TBD | TBD | TBD | TBD |
| DCP | overall | 213 | 19 | TBD | TBD | TBD | TBD |
| DCP | indoor | 207 | 13 | TBD | TBD | TBD | TBD |
| DCP | outdoor | 6 | 6 | TBD | TBD | TBD | TBD |
| AOD-Net | overall | 213 | 19 | TBD | TBD | TBD | TBD |
| AOD-Net | indoor | 207 | 13 | TBD | TBD | TBD | TBD |
| AOD-Net | outdoor | 6 | 6 | TBD | TBD | TBD | TBD |
| FFA-Net | overall | 213 | 19 | TBD | TBD | TBD | TBD |
| FFA-Net | indoor | 207 | 13 | TBD | TBD | TBD | TBD |
| FFA-Net | outdoor | 6 | 6 | TBD | TBD | TBD | TBD |

## Limitations

- The outdoor test set has only 6 images (6 groups), so outdoor rows are noisy.
- The test set has only 19 scene groups, so confidence intervals are wide.
- Scene grouping is approximate; some test rooms may have close copies in train.
- Stretching to 256x256 changes the aspect ratio and removes detail; scores at full
  resolution may differ.
- Our models use CLAHE/HLS preprocessing while DCP and external models get the raw
  image, so the inputs are not identical.
- External models are run outside this repository, with weights and settings chosen
  there, possibly trained on other data. Their scores depend on that setup.
- One training run per model; variation between runs is not measured.
- Timing depends on hardware. TensorFlow 2.11+ has no GPU support on native Windows,
  so times measured there are CPU times.
- PSNR and SSIM do not fully match how a person judges image quality.
