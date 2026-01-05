import fiftyone as fo
import fiftyone.zoo as foz

# Dataset name
dataset_name = "pixelforge_openimages_v1"

# Classes to download (Phase 1)
classes = [
"Person","Indoor","Outdoor","Product","Food","Animal","Vehicle","Architecture","Landscape"
]

# Max images per class
max_images = 100

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    label_types=[],              # we don't need bounding boxes
    classes=classes,
    max_samples=max_images * len(classes),
    shuffle=True,
    dataset_name=dataset_name
)

print("Download completed!")
print("Total images:", len(dataset))
