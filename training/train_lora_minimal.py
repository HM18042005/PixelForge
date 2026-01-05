"""
Simplified SDXL LoRA Training Script
Focuses on minimal training loop with skip-VAE encoding for faster testing
"""
import argparse
import logging
import os
import torch
import torch.nn.functional as F
from pathlib import Path

from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output_dir", default="sdxl-lora-output")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_steps", type=int, default=200)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 80)
    logger.info("SDXL LoRA Minimal Training Started")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Output: {args.output_dir}")
    
    # Step 1: Load UNet only
    logger.info("Step 1: Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    logger.info("✓ UNet loaded and frozen")
    
    # Step 2: Add LoRA
    logger.info("Step 2: Adding LoRA...")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"✓ LoRA added - Trainable params: {trainable:,}")
    
    # Move to device
    unet = unet.to(device)
    
    # Step 3: Setup optimizer
    logger.info("Step 3: Setting up optimizer...")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=args.learning_rate)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.num_train_steps)
    logger.info("✓ Optimizer ready")
    
    # Step 4: Create dummy data
    logger.info("Step 4: Creating dummy training data...")
    # Dummy latents (batch_size, channels=4, height=128, width=128) - 1024/8 = 128
    dummy_latents = torch.randn(args.batch_size, 4, 128, 128)
    # Dummy encoder hidden states (batch_size, seq_len=77, hidden_size=2048)
    dummy_embeddings = torch.randn(args.batch_size, 77, 2048)
    # Dummy add embeddings (batch_size, 1280)
    dummy_add_emb = torch.randn(args.batch_size, 1280)
    # Dummy time ids (batch_size, 6)
    dummy_time_ids = torch.randn(args.batch_size, 6)
    
    dataset = TensorDataset(dummy_latents, dummy_embeddings, dummy_add_emb, dummy_time_ids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"✓ Created {len(dataloader)} dummy batches")
    
    # Step 5: Training loop
    logger.info("Step 5: Starting training...")
    logger.info("=" * 80)
    
    unet.train()
    progress_bar = tqdm(total=args.num_train_steps)
    
    for step in range(args.num_train_steps):
        # Get dummy batch
        batch_idx = step % len(dataloader)
        latents, embeds, add_emb, time_ids = dataset[batch_idx]
        
        latents = latents.unsqueeze(0).to(device) if latents.dim() == 3 else latents.to(device)
        embeds = embeds.unsqueeze(0).to(device) if embeds.dim() == 2 else embeds.to(device)
        add_emb = add_emb.unsqueeze(0).to(device) if add_emb.dim() == 1 else add_emb.to(device)
        time_ids = time_ids.unsqueeze(0).to(device) if time_ids.dim() == 1 else time_ids.to(device)
        
        # Create dummy timesteps
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
        
        # Forward pass
        output = unet(
            latents,
            timesteps,
            encoder_hidden_states=embeds,
            added_cond_kwargs={"text_embeds": add_emb, "time_ids": time_ids}
        )
        
        # Dummy loss
        loss = output.sample.abs().mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if (step + 1) % 50 == 0:
            logger.info(f"Step {step + 1}/{args.num_train_steps} - Loss: {loss.item():.4f}")
    
    progress_bar.close()
    logger.info("=" * 80)
    
    # Step 6: Save
    logger.info("Step 6: Saving LoRA weights...")
    os.makedirs(args.output_dir, exist_ok=True)
    unwrapped_unet = unet if not hasattr(unet, 'module') else unet.module
    lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers=lora_state_dict,
    )
    logger.info(f"✓ LoRA saved to {args.output_dir}")
    logger.info("=" * 80)
    logger.info("✓ Training Complete!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
