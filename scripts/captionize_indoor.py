#!/usr/bin/env python3
"""Create per-image captions and a master captions.txt for indoor dataset images.

- Reads images from datasets/indoor/images (configurable).
- Writes one-line caption files alongside each image: <photo_name>.txt
- Rewrites a master captions.txt in the dataset root (filename<TAB>caption).
- Deterministic rule-based caption (no ML): indoor room with furniture and decor, realistic lighting.
"""
from pathlib import Path
import argparse

DEFAULT_CAPTION = "indoor room with furniture and decor, realistic lighting"
VALID_EXT = {".jpg", ".jpeg", ".png"}


def make_caption_files(images_dir: Path, caption_text: str) -> None:
    dataset_root = images_dir.parent
    captions_path = dataset_root / "captions.txt"

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXT])
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    lines = []
    for img in images:
        fname = img.name
        # Per-image caption file (same base name, .txt)
        txt_path = images_dir / (img.stem + ".txt")
        txt_path.write_text(caption_text, encoding="utf-8")
        lines.append(f"{fname}\t{caption_text}")

    captions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(images)} per-image caption files in {images_dir}")
    print(f"Wrote master captions file: {captions_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate per-image captions for indoor dataset")
    p.add_argument("--images-dir", type=Path, default=Path("datasets/indoor/images"), help="Directory containing images")
    p.add_argument("--caption", type=str, default=DEFAULT_CAPTION, help="Caption text to apply to all images")
    return p.parse_args()


def main():
    args = parse_args()
    make_caption_files(args.images_dir, args.caption)


if __name__ == "__main__":
    main()
