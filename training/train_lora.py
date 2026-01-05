#!/usr/bin/env python3
"""
Stable Diffusion 1.5 LoRA Fine-tuning Script
Trains LoRA adapters on custom image-caption pairs for PixelForge
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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


class ImageCaptionDataset(Dataset):
    """Dataset for image-caption pairs"""

    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        resolution: int = 512,
        center_crop: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.center_crop = center_crop

        # Load image-caption pairs
        self.pairs = []
        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    image_name, caption = parts[0], parts[1]
                    image_path = self.image_dir / image_name
                    if image_path.exists():
                        self.pairs.append((image_path, caption))

        logger.info(f"Loaded {len(self.pairs)} image-caption pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        image_path, caption = self.pairs[idx]

        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        
        # Resize maintaining aspect ratio then center crop
        image = image.resize(
            (self.resolution, self.resolution),
            Image.Resampling.LANCZOS,
        )
        
        if self.center_crop:
            left = (self.resolution - self.resolution) // 2
            top = (self.resolution - self.resolution) // 2
            image = image.crop(
                (left, top, left + self.resolution, top + self.resolution)
            )

        # Convert to tensor and normalize to [-1, 1]
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)

        return {
            "image": image,
            "caption": caption,
            "image_path": str(image_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion 1.5 with LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model ID from Hugging Face",
    )

    # Data arguments
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="datasets/raw_openimages",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="datasets/captions.txt",
        help="TSV file with image names and captions",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd15-lora-output",
        help="Output directory for checkpoints and weights",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank (r)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: same as rank)",
    )

    # Image arguments
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (height and width)",
    )

    # Optimization arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch"],
        help="Save checkpoint strategy",
    )

    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log loss every N steps",
    )

    return parser.parse_args()


def create_lora_config(args: argparse.Namespace) -> LoraConfig:
    """Create LoRA configuration"""
    lora_alpha = args.lora_alpha or args.lora_rank

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    return config


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting Stable Diffusion 1.5 LoRA Training")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Resolution: {args.resolution}x{args.resolution}")
    logger.info(f"Batch size: {args.train_batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 80)

    # Load models with cache
    logger.info("Loading models...")
    try:
        logger.info("  Loading text encoder...")
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_id, subfolder="text_encoder", cache_dir="./models"
        )
        logger.info("  ✓ Text encoder loaded")
        
        logger.info("  Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            args.model_id, subfolder="vae", cache_dir="./models"
        )
        logger.info("  ✓ VAE loaded")
        
        logger.info("  Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            args.model_id, subfolder="unet", cache_dir="./models"
        )
        logger.info("  ✓ UNet loaded")
        
        logger.info("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", use_fast=False, cache_dir="./models"
        )
        logger.info("  ✓ Tokenizer loaded")
        
        logger.info("  Loading scheduler...")
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.model_id, subfolder="scheduler", cache_dir="./models"
        )
        logger.info("  ✓ Scheduler loaded")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Make sure you have internet connection or models cached locally")
        raise

    # Freeze base models
    logger.info("Freezing base model parameters...")
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA
    logger.info("Adding LoRA adapters...")
    lora_config = create_lora_config(args)
    unet = get_peft_model(unet, lora_config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
    )

    # Load dataset
    logger.info("Loading dataset...")
    dataset = ImageCaptionDataset(
        image_dir=args.train_data_dir,
        captions_file=args.captions_file,
        resolution=args.resolution,
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Setup optimizer and scheduler
    logger.info("Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
    )

    num_training_steps = (
        len(train_dataloader) * args.num_train_epochs
    ) // args.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with accelerator
    logger.info("Preparing with accelerator...")
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move other models to device
    text_encoder = text_encoder.to(accelerator.device)
    vae = vae.to(accelerator.device)

    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 80)

    progress_bar = tqdm(
        total=num_training_steps,
        disable=not accelerator.is_main_process,
        desc="Training",
    )

    global_step = 0

    for epoch in range(args.num_train_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                # Encode images to latent space
                images = batch["image"].to(accelerator.device)
                images = images.to(vae.dtype)
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Encode text
                input_ids = tokenizer(
                    batch["caption"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Random timesteps
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ).long()

            # Add noise to latents (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise residual
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # Compute loss
            loss = F.mse_loss(model_pred, noise, reduction="mean")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        unet.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.logging_steps == 0:
                        loss_avg = loss.detach().item()
                        logger.info(
                            f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss_avg:.4f}"
                        )

                    # Save checkpoint
                    if (
                        args.save_strategy == "steps"
                        and global_step % args.checkpointing_steps == 0
                    ):
                        checkpoint_dir = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        unwrapped_unet = accelerator.unwrap_model(unet)
                        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=checkpoint_dir,
                            unet_lora_layers=lora_state_dict,
                        )
                        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    progress_bar.close()
    logger.info("=" * 80)

    # Save final weights
    logger.info("Saving final LoRA weights...")
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=lora_state_dict,
        )
        logger.info(f"LoRA weights saved to {args.output_dir}")

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    import numpy as np
    main()
