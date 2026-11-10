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

Measured on 2026-11-11, full test split (213 images, 19 scene groups, no `--limit`).
Saved copies: `results/results_test_2026-11-11.md`, `results/results_test_2026-11-11.csv`,
`results/results_per_image_test_2026-11-11.csv`.

```
python evaluate.py --model-path outputs/weights_all/models/unet/unet_best.keras \
                   --model-path outputs/weights_all/models/pix2pix/p2p_best.keras \
                   --compare-dcp --split test
```

Evaluation device: CPU only (AMD Ryzen 7 5800H, native Windows 11, TensorFlow 2.21,
Keras 3.15; no GPU on native Windows). Both models were loaded with `compile=False`
and used the preprocess mode from their sidecar json (`clahe_hls`).

| Method | Subset | Images | Groups | PSNR (group mean) [95% CI] | SSIM (group mean) [95% CI] | Params | ms/image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net (`unet_best`) | overall | 213 | 19 | 18.45 [17.23, 19.69] | 0.7098 [0.6430, 0.7729] | 31,379,075 | 1160.3 |
| U-Net (`unet_best`) | indoor | 207 | 13 | 19.91 [18.95, 20.97] | 0.7813 [0.7314, 0.8287] | 31,379,075 | 1160.3 |
| U-Net (`unet_best`) | outdoor | 6 | 6 | 15.26 [14.38, 16.28] | 0.5550 [0.4614, 0.6278] | 31,379,075 | 1160.3 |
| Pix2Pix (`p2p_best`) | overall | 213 | 19 | 17.69 [16.42, 18.96] | 0.6709 [0.6043, 0.7352] | 31,379,075 | 1137.2 |
| Pix2Pix (`p2p_best`) | indoor | 207 | 13 | 19.12 [17.96, 20.40] | 0.7383 [0.6821, 0.7942] | 31,379,075 | 1137.2 |
| Pix2Pix (`p2p_best`) | outdoor | 6 | 6 | 14.58 [13.79, 15.33] | 0.5251 [0.4401, 0.5973] | 31,379,075 | 1137.2 |
| DCP | overall | 213 | 19 | 13.35 [11.86, 15.06] | 0.6183 [0.5375, 0.7000] | 0 | 11.9 |
| DCP | indoor | 207 | 13 | 14.17 [12.23, 16.43] | 0.7159 [0.6567, 0.7765] | 0 | 11.9 |
| DCP | outdoor | 6 | 6 | 11.57 [10.23, 12.89] | 0.4069 [0.3130, 0.5007] | 0 | 11.9 |

Image means (not group means) are in the saved `results_test_2026-11-11.md`.
ms/image for our models is CPU time at batch size 1; it would be much lower on a GPU.
AOD-Net and FFA-Net have not been measured yet.

### Training facts (from `log.csv`)

| Model | Epochs run | Best epoch | Val PSNR at best | Val SSIM at best | Notes |
| --- | --- | --- | --- | --- | --- |
| U-Net | 22 (early stop) | 17 | 18.37 | 0.739 | train loss 0.0027 vs val loss 0.0190 at epoch 17 |
| Pix2Pix | 20 (all) | 15 | 18.31 | 0.727 | no early stop |

Both trained on Kaggle (Tesla T4 GPU, Keras 3.13), batch size 3. The `*_best.keras`
files are the epochs with the best val PSNR.

### Interpretation

- On PSNR, both trained models clearly beat DCP: overall and indoor CIs do not overlap
  (U-Net 18.45 [17.23, 19.69], Pix2Pix 17.69 [16.42, 18.96] vs DCP 13.35 [11.86, 15.06]).
- On SSIM, DCP is not clearly worse: its CIs overlap with both models overall and indoor.
  Indoor, DCP's SSIM (0.716) is close to Pix2Pix (0.738).
- U-Net is ahead of Pix2Pix on every point estimate, but every CI overlaps, so this
  benchmark does not show that one is better than the other.
- Outdoor numbers (6 images) are lower for all methods and too noisy to rank anything.
- Overlap of separate CIs is only a rough check; no paired test between methods was run.

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
- U-Net overfits heavily: at its best epoch the train loss is 0.0027 but the val loss is
  0.0190 (about 7x higher), and val loss stopped improving while train loss kept falling.
- Validation numbers were used to choose the best epoch (and for early stopping), so
  the val scores in the training table are optimistic. The test split was not used for
  any training or model choice.
- Timing depends on hardware. TensorFlow 2.11+ has no GPU support on native Windows,
  so times measured there are CPU times.
- PSNR and SSIM do not fully match how a person judges image quality.
