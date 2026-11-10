import json

import numpy as np

import config
from baselines.dcp import dehaze as dcp_dehaze
from evaluate import PSNR_CAP, score
from preprocess import prepare_input


def random_image(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_prepare_input_shape_and_dtype():
    out = prepare_input(random_image(480, 640))
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8


def test_split_groups_do_not_overlap():
    splits = json.loads(config.SPLIT_FILE.read_text(encoding="utf-8"))["splits"]
    groups = {name: {item["group_id"] for item in splits[name]} for name in ["train", "val", "test"]}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_dcp_keeps_shape():
    img = random_image(120, 160)
    out = dcp_dehaze(img)
    assert out.shape == img.shape


def test_psnr_of_same_image_is_capped():
    img = random_image(64, 64)
    psnr, ssim = score(img, img)
    assert psnr == PSNR_CAP == 100.0
    assert abs(ssim - 1.0) < 1e-6
