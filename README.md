# De-hazer

Removes haze from images and videos with a U-Net or a Pix2Pix model (TensorFlow/Keras),
then counts people in the result with OpenCV's HOG detector.

## Setup

```
git clone https://github.com/paulrounak/De-Haze.git
cd De-Haze
pip install -r requirements.txt
```

TensorFlow 2.11+ has no GPU support on native Windows, so training there runs on the CPU.

## Dataset

Put the images in `dataset/`:

- `Hazy_Full/`, `GT_Full/`: all 2036 pairs (`01_hazy.png` / `01_GT.png`, ...)
- `Hazy_Small/`, `GT_Small/`: the 55 outdoor pairs (IDs 01-55), used by `--small`

Files 56 and above are JPEG data with a `.png` name; `cv2.imread` reads them fine.

## Train

```
python train.py --model unet                    # 60 epochs, batch 3
python train.py --model pix2pix                 # 15 epochs, batch 3
python train.py --model unet --small --epochs 1 # quick check on 55 images
```

Other options: `--epochs`, `--batch-size`, `--preprocess clahe_hls|clahe|none`,
`--max-hours` (default 10), `--output-dir`, `--checkpoint-dir`, `--rebuild-split`.
Run `python train.py --help` for all of them.

A checkpoint is saved after every epoch in `outputs/checkpoints/<model>/`. Running the
same command again continues from it. To start over, use another `--checkpoint-dir`.
Training stops early when the validation loss has not improved for 5 epochs, or before
`--max-hours` would be passed.

Models are saved to `outputs/models/<model>/`: the final model as `.keras` and `.h5`,
the best one (lowest validation loss) as `<name>_best.keras`, each with a `.json`
file that records the preprocessing mode. `infer.py` reads this file so that inference
uses the same preprocessing as training. `log.csv` has one row per epoch.

`--small` is only for debugging. Do not report numbers from it.

## Train on Kaggle

1. `python kaggle/make_code_zip.py` creates `outputs/de-haze-code.zip` (code, split file,
   sample files; no images, weights or caches). Upload it as a Kaggle dataset.
2. Open `kaggle/train_kaggle.ipynb` on Kaggle, attach the code dataset and the images
   dataset, choose a GPU and use Save & Run All.
3. Download `weights_all.zip` from the notebook output.

## Run on an image or video

```
python infer.py --model-path outputs/models/unet/unet.keras --input sample_test_files/hazed_img.jpg
python infer.py --model-path outputs/models/unet/unet.keras --input sample_test_files/test_vid.mp4 --no-display
```

The output has the same size as the input and is saved in `outputs/`.
Options: `--output`, `--no-display`, `--no-detect`, `--detect-on dehazed|hazy`, `--preprocess`.
Press `q` to stop a video.

## How it works

1. **Preprocessing** (`preprocess.py`): CLAHE on the L channel (LAB), then L-40 / S+30 in HLS.
   This runs on the full-size hazy image, which is then resized to 256x256 with `INTER_AREA`.
   Ground-truth images are only resized. All images stay in BGR.
2. **Split** (`split.py`, `splits/split_v1.json`): many indoor images show the same room.
   Similar images are grouped (MobileNetV2 features, clustering at cosine distance 0.20) and
   each group goes entirely into train, val or test, so the test set does not contain copies
   of training rooms. Training always reads the saved JSON file; rebuild it only on purpose.
3. **Models** (`models.py`): U-Net with a sigmoid output. Pix2Pix uses the same U-Net as the
   generator and a 70x70 PatchGAN discriminator, with GAN loss + 100 x L1 loss.
4. **Detection** (`detect.py`): HOG people detector on the dehazed output (or on the input
   with `--detect-on hazy`).

Validation PSNR and SSIM are printed after every epoch.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | sizes, epochs, paths, split settings |
| `data.py` | loading, caching (`outputs/cache/`), `tf.data` pipeline |
| `preprocess.py` | CLAHE/HLS preprocessing |
| `models.py` | U-Net, PatchGAN, losses |
| `train.py` | training for both models |
| `infer.py` | image/video inference |
| `detect.py` | HOG human detection |
| `split.py` | builds `splits/split_v1.json` |

## Trained models

Google Drive: https://drive.google.com/drive/folders/1Uh8nfn4vCnb6XOZRrU2qhV-IaLTSqJNp?usp=sharing

## Sample input and output

### Sample input

![sample_input](https://github.com/user-attachments/assets/8f719c78-5f23-4f6f-870f-77f907416f16)

### Output: U-Net

![output_Unet](https://github.com/user-attachments/assets/27215d29-61e2-4128-9245-d5929a19fe36)

### Output: Pix2Pix

![output_Pix2Pix](https://github.com/user-attachments/assets/f80eaf93-91cf-470b-aa2b-79448e9746b2)
