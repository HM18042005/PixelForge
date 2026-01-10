
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS_PATH = "sd15-lora-output"

PROMPT = "a cinematic photo of window showing sunset, ultra detailed, realistic lighting"
NEGATIVE_PROMPT = "blurry, low quality, distorted, artifacts"

STEPS = 25
CFG_SCALE = 7.5
WIDTH = 768
HEIGHT = 512
SEED = 94


def generate_comparison():
    """Generate both normal and LoRA fine-tuned images for comparison"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 80)

    # Load base pipeline
    print("Loading Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    # Generate normal image (without LoRA)
    print("\n[1/2] Generating NORMAL image (without LoRA)...")
    print(f"Prompt: {PROMPT}")
    normal_image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=CFG_SCALE,
        generator=generator,
        width=WIDTH,
        height=HEIGHT
    ).images[0]
    
    normal_output = "output_normal.png"
    normal_image.save(normal_output)
    print(f"✓ Normal image saved as {normal_output}")
    
    # Load LoRA weights - fuse them directly into UNet 
    print(f"\n[2/2] Loading LoRA weights from {LORA_WEIGHTS_PATH}...")
    from safetensors.torch import load_file
    import os
    
    lora_file = os.path.join(LORA_WEIGHTS_PATH, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_file):
        print(f"Error: Could not find {lora_file}")
        print("Generating only the normal image...")
        return
    
    # Load and process LoRA weights
    lora_state_dict = load_file(lora_file)
    print(f"Loaded {len(lora_state_dict)} LoRA weight tensors")
    
    # Strip "base_model.model." prefix from keys for compatibility
    processed_lora = {}
    for key, value in lora_state_dict.items():
        new_key = key.replace("base_model.model.", "")
        processed_lora[new_key] = value
    
    # Merge LoRA weights into UNet (permanent fusion for this session)
    print("Fusing LoRA weights into UNet...")
    from peft import PeftModel, LoraConfig
    
    # Recreate LoRA config matching training
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    
    # Inject LoRA into UNet
    unet_lora = PeftModel(pipe.unet, lora_config)
    unet_lora.load_state_dict(processed_lora, strict=False)
    unet_lora.merge_and_unload()  # Fuse LoRA weights permanently
    
    pipe.unet = unet_lora.to(device)
    print("✓ LoRA weights fused successfully")
    
    # Reset generator with same seed for fair comparison
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    # Generate LoRA fine-tuned image
    print("\nGenerating LORA FINE-TUNED image...")
    print(f"Prompt: {PROMPT}")
    lora_image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=CFG_SCALE,
        generator=generator,
        width=WIDTH,
        height=HEIGHT
    ).images[0]
    
    lora_output = "output_lora.png"
    lora_image.save(lora_output)
    print(f"✓ LoRA image saved as {lora_output}")
    
    # Create side-by-side comparison
    print("\nCreating side-by-side comparison...")
    comparison_width = WIDTH * 2 + 20  # 20px padding between images
    comparison_height = HEIGHT + 60  # Extra space for labels
    
    comparison = Image.new('RGB', (comparison_width, comparison_height), 'white')
    
    # Paste images
    comparison.paste(normal_image, (0, 40))
    comparison.paste(lora_image, (WIDTH + 20, 40))
    
    # Add labels (simple text simulation using image manipulation)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    
    try:
        # Try to use a nicer font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    draw.text((WIDTH // 2 - 80, 10), "NORMAL (No LoRA)", fill='black', font=font)
    draw.text((WIDTH + WIDTH // 2 - 60, 10), "LORA FINE-TUNED", fill='black', font=font)
    
    comparison_output = "output_comparison.png"
    comparison.save(comparison_output)
    print(f"✓ Comparison image saved as {comparison_output}")
    
    print("\n" + "=" * 80)
    print("Generation complete!")
    print(f"\nFiles created:")
    print(f"  - {normal_output} (baseline)")
    print(f"  - {lora_output} (with LoRA)")
    print(f"  - {comparison_output} (side-by-side)")
    print("=" * 80)


if __name__ == "__main__":
    generate_comparison()
