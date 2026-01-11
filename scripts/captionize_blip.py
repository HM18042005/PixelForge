#!/usr/bin/env python3
"""Generate per-image captions using BLIP and write per-image .txt plus captions.txt.

Default dataset: datasets/indoor/images
Model: Salesforce/blip-image-captioning-large

Outputs:
- For each image: <stem>.txt next to the image
- Master captions file: captions.txt in the dataset root with lines: filename<TAB>caption

Usage examples:
  python scripts/captionize_blip.py
  python scripts/captionize_blip.py --images-dir datasets/indoor/images --model Salesforce/blip-image-captioning-large
"""
from pathlib import Path
import argparse

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm

DEFAULT_MODEL = "Salesforce/blip-image-captioning-large"
VALID_EXT = {".jpg", ".jpeg", ".png"}


def load_model(model_id: str, device: str):
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model


def caption_image(img_path: Path, processor, model, device: str, max_new_tokens: int = 40) -> str:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


def run(images_dir: Path, model_id: str, max_new_tokens: int = 40):
    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor, model = load_model(model_id, device)

    dataset_root = images_dir.parent
    captions_path = dataset_root / "captions.txt"

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXT])
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    lines = []
    for img in tqdm(images, desc="Captioning"):
        caption = caption_image(img, processor, model, device, max_new_tokens=max_new_tokens)
        txt_path = images_dir / (img.stem + ".txt")
        txt_path.write_text(caption, encoding="utf-8")
        lines.append(f"{img.name}\t{caption}")

    captions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(images)} captions to per-image txt files and {captions_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Caption images with BLIP and write per-image txt + captions.txt")
    p.add_argument("--images-dir", type=Path, default=Path("datasets/indoor/images"), help="Directory containing images")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="BLIP model id")
    p.add_argument("--max-new-tokens", type=int, default=40, help="Max new tokens for generation")
    return p.parse_args()


def main():
    args = parse_args()
    run(args.images_dir, args.model, args.max_new_tokens)


if __name__ == "__main__":
    main()
