import argparse
import os
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import subprocess

# =========================
# CONFIG
# =========================
BASE_DIR = Path("datasets/architecture")
IMAGES_DIR = BASE_DIR / "images"
CAPTIONS_FILE = BASE_DIR / "captions.txt"

TARGET_COUNT = 1200

KAGGLE_DATASETS = {
    "narrow_old": (
        "ashishjangra27/architecture-buildings-images",
        "narrow historic stone street with old buildings, realistic architecture"
    ),
    "modern_street": (
        "ryanholbrook/cityscapes",
        "modern city street with mid-rise buildings, realistic urban architecture"
    ),
    "wide_angle": (
        "dataclusterlabs/urban-scenes-dataset",
        "wide angle view of a city street, strong perspective depth, realistic cityscape"
    ),
    "facade_close": (
        "kmader/architectural-details",
        "close-up of building facade with windows and stone texture, realistic architecture"
    )
}

VALID_EXT = (".jpg", ".jpeg", ".png")

# =========================
# HELPERS
# =========================
def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def collect_images(src_dir):
    imgs = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                imgs.append(Path(root) / f)
    return imgs

def normalize_image(img_path, dest):
    try:
        img = Image.open(img_path).convert("RGB")
        img.save(dest, "JPEG", quality=92)
        return True
    except:
        return False

# =========================
# MAIN PIPELINE
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["kaggle"], default="kaggle")
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_index = 1
    captions = []

    for cat, (slug, caption) in KAGGLE_DATASETS.items():
        print(f"\n📥 Downloading {cat} from Kaggle: {slug}")
        tmp_dir = Path("tmp") / cat
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        run(f"kaggle datasets download -d {slug} -p {tmp_dir} --unzip")

        imgs = collect_images(tmp_dir)
        random.shuffle(imgs)

        take = TARGET_COUNT // len(KAGGLE_DATASETS)
        taken = 0

        for img_path in tqdm(imgs, desc=f"Processing {cat}", ncols=80):
            if taken >= take:
                break

            out_name = f"img_{image_index:04d}.jpg"
            out_path = IMAGES_DIR / out_name

            if normalize_image(img_path, out_path):
                captions.append(f"{out_name}\t{caption}")
                image_index += 1
                taken += 1

        shutil.rmtree(tmp_dir)

    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        for line in captions:
            f.write(line + "\n")

    print("\n✅ DATASET READY")
    print(f"Images: {image_index-1}")
    print(f"Captions: {CAPTIONS_FILE}")

if __name__ == "__main__":
    main()
