
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from safetensors.torch import load_file, save_file

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_INDOOR_PATH = "sd15-lora-indoor"
LORA_OUTPUT_PATH = "sd15-lora-output"

# Use an in-dataset caption for a fair LoRA comparison
PROMPT = "make a highly realistic image of an airport lobby photo realistic"
NEGATIVE_PROMPT = "blurry, low quality, distorted, artifacts"

STEPS = 50
CFG_SCALE = 7.5
WIDTH = 768
HEIGHT = 512
SEED = 94
LORA_SCALE = 1.5  # Default indoor LoRA scale
LORA_OUTPUT_SCALE = 1.0  # Default output LoRA scale


_PIPE: Optional[StableDiffusionPipeline] = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ADAPTERS = [
    ("indoor", LORA_INDOOR_PATH),
    ("output", LORA_OUTPUT_PATH),
]


@dataclass
class GenerationResult:
    normal: Image.Image
    lora: Image.Image
    comparison: Optional[Image.Image]
    metadata: Dict

    def images(self) -> Dict[str, Image.Image]:
        return {
            "normal": self.normal,
            "lora": self.lora,
            "comparison": self.comparison,
        }


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


def _load_pipeline(lora_scales: Dict[str, float]) -> StableDiffusionPipeline:
    global _PIPE

    if _PIPE is not None:
        _PIPE.set_adapters(
            [name for name, _ in _ADAPTERS],
            adapter_weights=[lora_scales.get(name, 0.0) for name, _ in _ADAPTERS],
        )
        return _PIPE

    dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.to(_DEVICE)
    pipe.enable_attention_slicing()

    for adapter_name, adapter_path in _ADAPTERS:
        lora_dir, weight_name = _prepare_lora_weights(adapter_path)
        pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name=adapter_name)

    pipe.set_adapters(
        [name for name, _ in _ADAPTERS],
        adapter_weights=[lora_scales.get(name, 0.0) for name, _ in _ADAPTERS],
    )

    _PIPE = pipe
    return _PIPE


def _make_comparison(normal_image: Image.Image, lora_image: Image.Image, width: int, height: int) -> Image.Image:
    comparison_width = width * 2 + 20
    comparison_height = height + 60

    comparison = Image.new("RGB", (comparison_width, comparison_height), "white")
    comparison.paste(normal_image, (0, 40))
    comparison.paste(lora_image, (width + 20, 40))

    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(comparison)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    draw.text((width // 2 - 80, 10), "NORMAL (No LoRA)", fill="black", font=font)
    draw.text((width + width // 2 - 60, 10), "LORA FINE-TUNED", fill="black", font=font)

    return comparison


def generate_images(
    prompt: str = PROMPT,
    negative_prompt: str = NEGATIVE_PROMPT,
    steps: int = STEPS,
    guidance_scale: float = CFG_SCALE,
    width: int = WIDTH,
    height: int = HEIGHT,
    seed: int = SEED,
    lora_scales: Optional[Dict[str, float]] = None,
    include_comparison: bool = True,
) -> GenerationResult:
    if lora_scales is None:
        lora_scales = {"indoor": LORA_SCALE, "output": LORA_OUTPUT_SCALE}

    pipe = _load_pipeline(lora_scales=lora_scales)

    pipe.set_adapters([name for name, _ in _ADAPTERS], adapter_weights=[0.0] * len(_ADAPTERS))
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    normal_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=width,
        height=height,
    ).images[0]

    pipe.set_adapters(
        [name for name, _ in _ADAPTERS],
        adapter_weights=[lora_scales.get(name, 0.0) for name, _ in _ADAPTERS],
    )
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    lora_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=width,
        height=height,
    ).images[0]

    comparison_image = _make_comparison(normal_image, lora_image, width, height) if include_comparison else None

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "seed": seed,
        "lora_scales": lora_scales,
        "device": _DEVICE,
    }

    return GenerationResult(
        normal=normal_image,
        lora=lora_image,
        comparison=comparison_image,
        metadata=metadata,
    )


def save_generation(result: GenerationResult, output_dir: str = "."):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normal_output = output_path / "output_normal.png"
    lora_output = output_path / "output_lora.png"
    comparison_output = output_path / "output_comparison.png"

    result.normal.save(normal_output)
    result.lora.save(lora_output)
    if result.comparison:
        result.comparison.save(comparison_output)

    print("=" * 80)
    print("Generation complete!")
    print(f"Device: {_DEVICE}")
    print(f"Prompt: {result.metadata['prompt']}")
    print("Files created:")
    print(f"  - {normal_output} (baseline)")
    print(f"  - {lora_output} (with LoRA)")
    if result.comparison:
        print(f"  - {comparison_output} (side-by-side)")
    print("=" * 80)


if __name__ == "__main__":
    save_generation(generate_images())
