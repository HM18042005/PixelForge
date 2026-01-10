#!/usr/bin/env python3
"""Extract indoor/archive.zip, filter images >=512px, convert to JPG, and write captions."""
from pathlib import Path
from PIL import Image
import zipfile
import shutil

ROOT = Path("datasets/indoor")
ZIP_PATH = ROOT / "archive.zip"
IMAGES_DIR = ROOT / "images"
CAPTIONS_PATH = ROOT / "captions.txt"
TEMP_EXTRACT = ROOT / "extracted"

VALID_EXT = {".jpg", ".jpeg", ".png"}
CAPTION_TEXT = "indoor room with furniture and decor, realistic lighting"


def next_index(images_dir: Path) -> int:
    existing = sorted(images_dir.glob("img_*.jpg"))
    if not existing:
        return 1
    try:
        return int(existing[-1].stem.split("_")[-1]) + 1
    except Exception:
        return len(existing) + 1


def main():
    if not ZIP_PATH.exists():
        raise SystemExit(f"zip not found: {ZIP_PATH}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_EXTRACT.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(TEMP_EXTRACT)

    idx = next_index(IMAGES_DIR)
    captions = []
    kept = 0

    for path in TEMP_EXTRACT.rglob("*"):
        if path.suffix.lower() not in VALID_EXT:
            continue
        try:
            with Image.open(path) as im:
                if min(im.size) < 512:
                    continue
                rgb = im.convert("RGB")
                fname = f"img_{idx:04d}.jpg"
                dest = IMAGES_DIR / fname
                rgb.save(dest, quality=92)
                captions.append(f"{fname}\t{CAPTION_TEXT}")
                idx += 1
                kept += 1
        except Exception:
            continue

    mode = "a" if CAPTIONS_PATH.exists() else "w"
    with CAPTIONS_PATH.open(mode, encoding="utf-8") as f:
        for line in captions:
            f.write(line + "\n")

    print(f"Kept {kept} images into {IMAGES_DIR}")
    print(f"Wrote {len(captions)} captions to {CAPTIONS_PATH}")

    shutil.rmtree(TEMP_EXTRACT)
    print("Cleaned up temporary extraction directory")


if __name__ == "__main__":
    main()
