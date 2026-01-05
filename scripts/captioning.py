import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

IMAGE_DIR = "datasets/raw_openimages"
CAPTION_FILE = "datasets/captions.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
).to(device)


with open(CAPTION_FILE, "w", encoding="utf-8") as f:
    for img_name in sorted(os.listdir(IMAGE_DIR)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        image = Image.open(img_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)

        # Optional but recommended
        caption = caption + ", photo realistic"

        f.write(f"{img_name}\t{caption}\n")
        print(img_name, "→", caption)

print("✅ Captioning completed!")
