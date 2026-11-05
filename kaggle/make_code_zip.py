"""Make outputs/de-haze-code.zip with only the code needed on Kaggle.

    python kaggle/make_code_zip.py

Included: all .py files (root, baselines/, kaggle/), splits/split_v1.json,
requirements.txt and sample_test_files/. Nothing else, so .venv, _old/, outputs/,
.git, __pycache__, model weights and the dataset images are never added.
Upload the zip on Kaggle as a new dataset (or a new version of the code dataset).
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "de-haze-code.zip"

files = (sorted(ROOT.glob("*.py")) + sorted((ROOT / "baselines").glob("*.py"))
         + sorted((ROOT / "kaggle").glob("*.py"))
         + [ROOT / "splits" / "split_v1.json", ROOT / "requirements.txt"]
         + sorted(p for p in (ROOT / "sample_test_files").iterdir() if p.is_file()))

OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        name = f.relative_to(ROOT).as_posix()
        z.write(f, name)
        print(f"  {name}  ({f.stat().st_size / 1024:.0f} KB)")
print(f"{len(files)} files -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
