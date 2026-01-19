"""CLI to run PixelForge generation with all LoRA adapters enabled.

Example:
    python scripts/generatecli.py --prompt "a cozy indoor cafe" --lora-indoor-scale 1.2 --lora-output-scale 0.8
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from generate import (
    CFG_SCALE,
    HEIGHT,
    LORA_OUTPUT_SCALE,
    LORA_SCALE,
    NEGATIVE_PROMPT,
    PROMPT,
    SEED,
    STEPS,
    WIDTH,
    generate_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with PixelForge LoRA adapters")
    parser.add_argument("--prompt", default=PROMPT, help="Positive prompt text")
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT, help="Negative prompt text")
    parser.add_argument("--steps", type=int, default=STEPS, help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=CFG_SCALE, help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=WIDTH, help="Output width (must be multiple of 8)")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Output height (must be multiple of 8)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--lora-indoor-scale", type=float, default=LORA_SCALE, help="Scale for indoor LoRA adapter")
    parser.add_argument("--lora-output-scale", type=float, default=LORA_OUTPUT_SCALE, help="Scale for output LoRA adapter")
    parser.add_argument("--no-comparison", action="store_true", help="Skip side-by-side comparison image")
    parser.add_argument("--out-dir", default="cli_outputs", help="Directory to save generated images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    result = generate_images(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        seed=args.seed,
        lora_scales={
            "indoor": args.lora_indoor_scale,
            "output": args.lora_output_scale,
        },
        include_comparison=not args.no_comparison,
    )

    normal_path = run_dir / "normal.png"
    lora_path = run_dir / "lora.png"
    comparison_path = run_dir / "comparison.png"

    result.normal.save(normal_path)
    result.lora.save(lora_path)
    if result.comparison:
        result.comparison.save(comparison_path)

    print("=== PixelForge GenerateCLI ===")
    print(f"Output directory: {run_dir}")
    print(f"Prompt: {args.prompt}")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}, Guidance: {args.guidance_scale}")
    print("LoRA scales:")
    print(f"  indoor: {args.lora_indoor_scale}")
    print(f"  output: {args.lora_output_scale}")
    print("Saved files:")
    print(f"  baseline: {normal_path}")
    print(f"  lora:     {lora_path}")
    if result.comparison:
        print(f"  compare:  {comparison_path}")
    print("Generation complete.")


if __name__ == "__main__":
    main()
