from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Image size used by both models
IMG_SIZE = (256, 256)          # (width, height) for cv2.resize
INPUT_SHAPE = (256, 256, 3)

# Training
UNET_EPOCHS = 60
UNET_BATCH_SIZE = 3
P2P_EPOCHS = 15
P2P_BATCH_SIZE = 3
SEED = 42
CHECKPOINT_EVERY = 5           # Pix2Pix: save a checkpoint every N epochs

# Preprocessing of hazy images: "clahe_hls" (original), "clahe" or "none"
PREPROCESS_MODE = "clahe_hls"
PREPROCESS_MODES = ["clahe_hls", "clahe", "none"]

# Folders
HAZY_FULL_DIR = ROOT / "dataset" / "Hazy_Full"
GT_FULL_DIR = ROOT / "dataset" / "GT_Full"
HAZY_SMALL_DIR = ROOT / "dataset" / "Hazy_Small"
GT_SMALL_DIR = ROOT / "dataset" / "GT_Small"
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
MODELS_DIR = OUTPUTS_DIR / "models"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"

# Train/val/test split (see split.py)
SPLIT_FILE = ROOT / "splits" / "split_v1.json"
EMBED_MODEL = "MobileNetV2"
EMBED_WEIGHTS = "imagenet"
EMBED_BATCH = 32
LINKAGE = "average"
# 0.20 chosen after a leakage sweep: 0.15 left ~58% of val/test images
# closer than 0.14 to a train image; 0.20 leaves ~10% (about 4% in test).
DISTANCE_THRESHOLD = 0.20
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
MAX_GROUP_SIZE_VAL_TEST = 30   # bigger groups always go to train
