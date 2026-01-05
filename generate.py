
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_PATH = "pixelforge_output.png"

PROMPT = "a cinematic photo of a city at sunset, ultra detailed, realistic lighting"
NEGATIVE_PROMPT = "blurry, low quality, distorted, artifacts"

STEPS = 25
CFG_SCALE = 7.5
WIDTH = 768
HEIGHT = 512
SEED = 94


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    pipe.to(device)

    # SD 1.5 memory optimizations
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=device).manual_seed(SEED)

    print("Generating image...")
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=CFG_SCALE,
        generator=generator
    ).images[0]

    image.save(OUTPUT_PATH)
    print(f"Image saved as {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
