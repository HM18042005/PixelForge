
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file, save_file

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS_PATH = "sd15-lora-output"

# Use an in-dataset caption for a fair LoRA comparison
PROMPT = "aa realistic city skyline with tall residential and office buildings, wide-angle view, high detail"
NEGATIVE_PROMPT = "blurry, low quality, distorted, artifacts"

STEPS = 50
CFG_SCALE = 7.5
WIDTH = 768
HEIGHT = 512
SEED = 94
LORA_SCALE = 1.5  # Increase to amplify LoRA effect (e.g., 1.0–2.0)


def _prepare_lora_weights(lora_dir: str, weight_name: str = "pytorch_lora_weights.safetensors"):
    """Strip legacy prefixes from LoRA keys so diffusers can load them.

    Normalizes old PEFT exports by collapsing prefixes to `unet.` when
    appropriate (e.g., `unet.base_model.model.` -> `unet.`) so keys align
    with diffusers UNet modules.
    """
    lora_dir_path = Path(lora_dir)
    src_path = lora_dir_path / weight_name
    if not src_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {src_path}")

    state_dict = load_file(src_path)
    remapped = {}
    stripped_any = False

    mapping = [
        ("unet.base_model.model.", "unet."),
        ("unet.model.", "unet."),
        ("unet.base_model.", "unet."),
        ("base_model.model.", "unet."),
        ("model.", "unet."),
    ]

    for key, value in state_dict.items():
        new_key = key
        for prefix, replacement in mapping:
            if new_key.startswith(prefix):
                new_key = replacement + new_key[len(prefix):]
                stripped_any = True
                break
        remapped[new_key] = value

    if not stripped_any:
        return lora_dir_path, weight_name

    fixed_name = "pytorch_lora_weights_stripped.safetensors"
    fixed_path = lora_dir_path / fixed_name
    save_file(remapped, fixed_path)
    return lora_dir_path, fixed_name


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
    
    # Load LoRA weights as adapter and apply adjustable scale (no fusion)
    print(f"\n[2/2] Loading LoRA weights from {LORA_WEIGHTS_PATH} with scale={LORA_SCALE}...")
    try:
        lora_dir, weight_name = _prepare_lora_weights(LORA_WEIGHTS_PATH)
        pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name="pixelforge")
        pipe.set_adapters(["pixelforge"], adapter_weights=[LORA_SCALE])
        print("✓ LoRA weights loaded and scaled")
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        print("Generating only the normal image...")
        return
    
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
