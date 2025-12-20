from __future__ import annotations

import sys
from pathlib import Path

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("The 'fiftyone' library is required to download Open Images V7.")
    print("Please install it using: pip install fiftyone")
    sys.exit(1)

DATASET_NAME = "open-images-v7"
DEFAULT_DEST = Path("datasets") / "open-images-v7"

# Categories mentioned in README:
# Person, Indoor, Outdoor, Product, Food, Animal, Vehicle, Architecture, and Landscape
# Mapping to Open Images V7 classes (approximate):
CLASSES = [
    "Person",
    "Food",
    "Animal",
    "Vehicle",
    "Building",  # Architecture
    "Landscape", # Might not exist directly, checking for generic
    "Plant",     # Outdoor/Landscape
    "Furniture", # Indoor
]

MAX_SAMPLES = 100  # Limit to 100 samples per run for safety

def download_dataset(dest: Path = DEFAULT_DEST, max_samples: int = MAX_SAMPLES) -> None:
    dest = dest.resolve()
    print(f"Downloading Open Images V7 subset to {dest}...")
    
    # Load the dataset from the zoo
    # This will download the dataset if not already present in fiftyone's cache
    # and then we can export it or just use it.
    # We use 'validation' split for a smaller download test.
    dataset = foz.load_zoo_dataset(
        DATASET_NAME,
        split="validation",
        label_types=["detections"],
        classes=CLASSES,
        max_samples=max_samples,
        shuffle=True,
        dataset_name="pixel-forge-open-images",
    )

    # Export to a directory in a standard format (e.g., COCO or generic)
    # Here we just ensure the images are where we expect or export them.
    # For simplicity, we'll export to the destination folder.
    dataset.export(
        export_dir=str(dest),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )
    
    print(f"Dataset exported to {dest}")

def main() -> None:
    download_dataset()

if __name__ == "__main__":
    main()
