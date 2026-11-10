# Results (test split)

- 213 of 213 test images, 19 scene groups.
- **Outdoor rows are noisy: only 6 outdoor images.**
- DCP is run on the raw hazy image resized with INTER_AREA, without the CLAHE/HLS preprocessing used by our models.
- PSNR/SSIM: tf.image, max_val 1.0, on uint8 outputs vs the ground truth stretched to 256x256 (INTER_AREA). Infinite PSNR is capped at 100 dB.
- Group mean = mean of per-scene-group means. 95% CI: bootstrap over scene groups (2000 resamples, seed 42).
- ms/image: batch size 1, after 3 warm-up runs, includes NumPy conversion.

| Method | Subset | Images | Groups | PSNR image mean | PSNR group mean [95% CI] | SSIM image mean | SSIM group mean [95% CI] | Params | ms/image | Device |
|---|---|---|---|---|---|---|---|---|---|---|
| unet_best | overall | 213 | 19 | 19.98 | 18.45 [17.23, 19.69] | 0.7825 | 0.7098 [0.6430, 0.7729] | 31,379,075 | 1160.3 | CPU |
| unet_best | indoor | 207 | 13 | 20.12 | 19.91 [18.95, 20.97] | 0.7891 | 0.7813 [0.7314, 0.8287] | 31,379,075 | 1160.3 | CPU |
| unet_best | outdoor | 6 | 6 | 15.26 | 15.26 [14.38, 16.28] | 0.5550 | 0.5550 [0.4614, 0.6278] | 31,379,075 | 1160.3 | CPU |
| p2p_best | overall | 213 | 19 | 19.20 | 17.69 [16.42, 18.96] | 0.7419 | 0.6709 [0.6043, 0.7352] | 31,379,075 | 1137.2 | CPU |
| p2p_best | indoor | 207 | 13 | 19.34 | 19.12 [17.96, 20.40] | 0.7482 | 0.7383 [0.6821, 0.7942] | 31,379,075 | 1137.2 | CPU |
| p2p_best | outdoor | 6 | 6 | 14.58 | 14.58 [13.79, 15.33] | 0.5251 | 0.5251 [0.4401, 0.5973] | 31,379,075 | 1137.2 | CPU |
| DCP | overall | 213 | 19 | 13.89 | 13.35 [11.86, 15.06] | 0.6977 | 0.6183 [0.5375, 0.7000] | 0 | 11.9 | CPU (NumPy/OpenCV) |
| DCP | indoor | 207 | 13 | 13.95 | 14.17 [12.23, 16.43] | 0.7061 | 0.7159 [0.6567, 0.7765] | 0 | 11.9 | CPU (NumPy/OpenCV) |
| DCP | outdoor | 6 | 6 | 11.57 | 11.57 [10.23, 12.89] | 0.4069 | 0.4069 [0.3130, 0.5007] | 0 | 11.9 | CPU (NumPy/OpenCV) |
