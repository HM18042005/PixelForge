import os

# Paths
IMAGE_DIR = "datasets/raw_openimages"
CAPTIONS_FILE = "datasets/captions.txt"

# Read global captions file
with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

count = 0

for line in lines:
    line = line.strip()
    if not line:
        continue

    # Split only on first tab or space sequence
    parts = line.split("\t", 1)
    if len(parts) != 2:
        print("⚠️ Skipping malformed line:", line)
        continue

    image_name, caption = parts
    image_path = os.path.join(IMAGE_DIR, image_name)

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_name}")
        continue

    # Create caption file with same base name
    base_name = os.path.splitext(image_name)[0]
    caption_path = os.path.join(IMAGE_DIR, base_name + ".txt")

    with open(caption_path, "w", encoding="utf-8") as cap_file:
        cap_file.write(caption)

    count += 1

print(f"✅ Created {count} caption files successfully!")
