#!/usr/bin/env python3
"""
Stable Diffusion 1.5 LoRA Fine-tuning Script (Demo Version)
Trains LoRA adapters with simulated data when models aren't available
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_lora.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion 1.5 with LoRA"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd15-lora-output",
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=50,
        help="Number of training steps (for demo)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(project_config=project_config)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting Stable Diffusion 1.5 LoRA Training (Demo)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 80)

    # Load models
    logger.info("Loading Stable Diffusion 1.5 models...")
    try:
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_id, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", use_fast=False
        )
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.model_id, subfolder="scheduler"
        )
        logger.info("✓ All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Check internet connection")
        logger.info("2. Try: huggingface-cli login")
        logger.info("3. Pre-download: huggingface-cli download runwayml/stable-diffusion-v1-5")
        return

    # Freeze base models
    logger.info("Freezing base model parameters...")
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA
    logger.info("Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(
        f"✓ Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
    )

    # Setup optimizer
    logger.info("Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_training_steps,
    )

    # Prepare with accelerator
    logger.info("Preparing models with accelerator...")
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    text_encoder = text_encoder.to(accelerator.device)
    vae = vae.to(accelerator.device)
    logger.info("✓ Models ready for training")

    # Training loop with dummy data
    logger.info("Starting training loop...")
    logger.info("=" * 80)

    progress_bar = tqdm(total=args.num_training_steps, desc="Training")
    unet.train()

    for step in range(args.num_training_steps):
        # Create dummy batch
        batch_size = args.batch_size
        
        # Dummy latents (4 channels, 64x64 resolution)
        latents = torch.randn(batch_size, 4, 64, 64, device=accelerator.device)
        
        # Dummy encoder hidden states (77 sequence length, 768 hidden size)
        encoder_hidden_states = torch.randn(
            batch_size, 77, 768, device=accelerator.device
        )
        
        # Dummy noise
        noise = torch.randn_like(latents)
        
        # Dummy timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=accelerator.device,
        ).long()

        # Forward pass
        model_pred = unet(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        # Loss
        loss = F.mse_loss(model_pred, noise, reduction="mean")

        # Backward
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.detach().item():.4f}"})

        if (step + 1) % 10 == 0:
            logger.info(f"Step {step + 1}/{args.num_training_steps} - Loss: {loss.detach().item():.4f}")

    progress_bar.close()
    logger.info("=" * 80)

    # Save weights
    logger.info("Saving LoRA weights...")
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=lora_state_dict,
        )
        logger.info(f"✓ LoRA weights saved to {args.output_dir}/pytorch_lora_weights.safetensors")

    logger.info("=" * 80)
    logger.info("✓ Training completed successfully!")
    logger.info("=" * 80)
    logger.info(f"\nTo use trained LoRA weights:")
    logger.info(f"  from diffusers import StableDiffusionPipeline")
    logger.info(f"  pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')")
    logger.info(f"  pipe.load_lora_weights('{args.output_dir}')")


if __name__ == "__main__":
    main()
