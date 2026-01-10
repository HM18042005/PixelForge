#!/usr/bin/env python3
"""
Architecture-focused data collection pipeline for Stable Diffusion LoRA.

Features
- Source flag: Open Images V7 or Kaggle
- Category mix totaling ~1.2k–1.5k images
- Size filter: discard images smaller than 512x512
- Saves RGB JPGs to datasets/architecture/images and captions to datasets/architecture/captions.txt
- Deterministic, concrete captions per category

Requirements
- Python 3.10
- For Open Images: pip install openimages
- For Kaggle: kaggle API configured (KAGGLE_USERNAME/KAGGLE_KEY) and dataset slugs provided
"""

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image

# Output locations
DATASET_ROOT = Path("datasets/architecture")
IMAGES_DIR = DATASET_ROOT / "images"
CAPTIONS_PATH = DATASET_ROOT / "captions.txt"

# Category config
@dataclass
class Category:
    name: str
    target: int
    labels: List[str]
    caption: str


CATEGORIES: List[Category] = [
    Category(
        name="narrow_old",
        target=480,
        labels=["Alley", "Street", "Facade", "Building"],
        caption="narrow historic stone street with old buildings, realistic architecture",
    ),
    Category(
        name="modern_street",
        target=330,
        labels=["Street", "Skyscraper", "Office building", "Apartment building"],
        caption="modern city street with mid-rise buildings, glass and concrete facades, urban architecture, daylight",
    ),
    Category(
        name="wide_angle",
        target=280,
        labels=["City", "Downtown", "Plaza", "Intersection"],
        caption="wide angle city street intersection with modern buildings, strong perspective depth",
    ),
    Category(
        name="facade_close",
        target=140,
        labels=["Facade", "Window", "Door", "Balcony"],
        caption="close-up of building facade with stone walls and windows, architectural detail, realistic texture",
    ),
    Category(
        name="negative_control",
        target=140,
        labels=["Tree", "Park", "Garden", "Sidewalk"],
        caption="urban street with trees and buildings in background, realistic city environment",
    ),
]

# Helpers

def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def next_index() -> int:
    existing = sorted(IMAGES_DIR.glob("img_*.jpg"))
    if not existing:
        return 1
    last = existing[-1].stem.split("_")[-1]
    try:
        return int(last) + 1
    except ValueError:
        return len(existing) + 1


def save_image(src_path: Path, dest_path: Path) -> bool:
    try:
        with Image.open(src_path) as img:
            if min(img.size) < 512:
                return False
            rgb = img.convert("RGB")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(dest_path, format="JPEG", quality=92)
        return True
    except Exception:
        return False


def append_caption(lines: List[str], filename: str, caption: str):
    lines.append(f"{filename}\t{caption}")


# Open Images pipeline

def download_openimages(category: Category, limit: int, tmpdir: Path):
    try:
        from openimages.download import download_dataset
    except ImportError:
        raise RuntimeError("Please install 'openimages' package: pip install openimages")

    out_dir = tmpdir / category.name
    out_dir.mkdir(parents=True, exist_ok=True)
    download_dataset(
        dataset_dir=str(out_dir),
        labels=category.labels,
        limit=limit,
        image_size=1024,
        format="jpg",
        retries=3,
    )
    return out_dir


# Kaggle pipeline

def download_kaggle(slug: str, tmpdir: Path) -> Path:
    out_dir = tmpdir / slug.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"]
    subprocess.run(cmd, check=True)
    return out_dir


# Dispatcher

def collect_images(source: str, kaggle_slugs: Dict[str, str]) -> None:
    ensure_dirs()
    start_idx = next_index()
    caption_lines: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for cat in CATEGORIES:
            needed = cat.target
            collected = 0
            if source == "openimages":
                cat_dir = download_openimages(cat, limit=needed * 2, tmpdir=tmpdir)
            elif source == "kaggle":
                if cat.name not in kaggle_slugs:
                    raise ValueError(f"Missing Kaggle slug for category {cat.name}")
                cat_dir = download_kaggle(kaggle_slugs[cat.name], tmpdir=tmpdir)
            else:
                raise ValueError("source must be 'openimages' or 'kaggle'")

            for img_path in cat_dir.rglob("*.jpg"):
                if collected >= needed:
                    break
                filename = f"img_{start_idx:04d}.jpg"
                dest = IMAGES_DIR / filename
                if save_image(img_path, dest):
                    append_caption(caption_lines, filename, cat.caption)
                    start_idx += 1
                    collected += 1

            print(f"{cat.name}: collected {collected}/{needed}")
            if collected < needed:
                print(f"Warning: category {cat.name} under target; rerun to top up.")

    # Write captions
    mode = "a" if CAPTIONS_PATH.exists() else "w"
    with CAPTIONS_PATH.open(mode, encoding="utf-8") as f:
        for line in caption_lines:
            f.write(line + "\n")
    print(f"Wrote {len(caption_lines)} captions to {CAPTIONS_PATH}")

    # Validate counts
    total = len(list(IMAGES_DIR.glob("img_*.jpg")))
    if not (1200 <= total <= 1500):
        print(f"Warning: total images {total} outside desired range (1200–1500)")
    else:
        print(f"Total images now: {total} (within target range)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Architecture LoRA dataset collector")
    parser.add_argument(
        "--source",
        type=str,
        choices=["openimages", "kaggle"],
        default="openimages",
        help="Data source",
    )
    parser.add_argument(
        "--kaggle-slug",
        action="append",
        default=[],
        help="Mapping entries category=slug (e.g., narrow_old=username/dataset)",
    )
    return parser.parse_args()


def parse_kaggle_mapping(entries: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for ent in entries:
        if "=" not in ent:
            continue
        key, val = ent.split("=", 1)
        mapping[key.strip()] = val.strip()
    return mapping


def main():
    args = parse_args()
    kaggle_map = parse_kaggle_mapping(args.kaggle_slug)
    collect_images(source=args.source, kaggle_slugs=kaggle_map)


if __name__ == "__main__":
    main()
