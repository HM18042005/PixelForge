# import torch
# from diffusers import StableDiffusionPipeline
# from PIL import Image
#
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
# OUTPUT_PATH = "pixelforge_output.png"
#
# PROMPT = "a cinematic photo of a human at sunset, ultra detailed, realistic lighting"
# NEGATIVE_PROMPT = "blurry, low quality, distorted, artifacts"
#
# STEPS = 25
# CFG_SCALE = 12
# WIDTH = 1080
# HEIGHT = 720
# SEED = 96
#
#
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#
#     pipe = StableDiffusionPipeline.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.float16,
#         safety_checker=None
#     )
#
#     pipe.to(device)
#
#     # Memory optimizations (critical for 8GB VRAM)
#     pipe.enable_attention_slicing()
#
#     generator = torch.Generator(device=device).manual_seed(SEED)
#
#     print("Generating image...")
#     image = pipe(
#         prompt=PROMPT,
#         negative_prompt=NEGATIVE_PROMPT,
#         num_inference_steps=STEPS,
#         guidance_scale=CFG_SCALE,
#         width=WIDTH,
#         height=HEIGHT,
#         generator=generator
#     ).images[0]
#
#     image.save(OUTPUT_PATH)
#     print(f"Image saved as {OUTPUT_PATH}")
#
#
# if __name__ == "__main__":
#     main()



import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_PATH = "pixelforge_output.png"

PROMPT = "a cinematic photo of a city at sunset, ultra detailed, realistic lighting"
NEGATIVE_PROMPT = "blurry, low quality, distorted, artifacts"

STEPS = 25
CFG_SCALE = 7.5   # SDXL prefers lower CFG
WIDTH = 1024
HEIGHT = 1024
SEED = 96


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None
    )

    pipe.to(device)

    # SDXL memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

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
