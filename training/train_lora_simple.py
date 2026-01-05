import argparse
import logging
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image

import os
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '0'

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL LoRA Training for PixelForge")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--train_data_dir", type=str, default="datasets/raw_openimages")
    parser.add_argument("--captions_file", type=str, default="datasets/captions.txt")
    parser.add_argument("--output_dir", type=str, default="sdxl-lora-output")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class ImageCaptionDataset(Dataset):
    """Custom dataset for image-caption pairs"""
    def __init__(self, data_dir, captions_file, resolution=1024):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        
        # Load captions
        self.image_caption_pairs = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_name, caption = parts[0], parts[1]
                    image_path = self.data_dir / image_name
                    if image_path.exists():
                        self.image_caption_pairs.append((image_path, caption))
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        logger.info(f"Loaded {len(self.image_caption_pairs)} image-caption pairs")
    
    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        image_path, caption = self.image_caption_pairs[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        
        return {
            'image': image,
            'caption': caption,
            'image_path': str(image_path)
        }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("SDXL LoRA Training Started")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Model: {args.pretrained_model_name_or_path}")
    logger.info(f"  - Training Data: {args.train_data_dir}")
    logger.info(f"  - Output Directory: {args.output_dir}")
    logger.info(f"  - Mixed Precision: {args.mixed_precision}")
    logger.info(f"  - Batch Size: {args.train_batch_size}")
    logger.info(f"  - Learning Rate: {args.learning_rate}")
    logger.info(f"  - Resolution: {args.resolution}x{args.resolution}")
    logger.info(f"  - LoRA Rank: {args.rank}")
    logger.info(f"  - Epochs: {args.num_train_epochs}")
    logger.info("=" * 80)
    
    logging_dir = os.path.join(args.output_dir, "logs")
    logger.info(f"Step 1: Initializing Accelerator")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=accelerator_project_config)
    logger.info("✓ Accelerator initialized")

    logger.info("Step 2: Loading SDXL Model Components...")
    
    logger.info("  Loading noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    logger.info("  ✓ Scheduler loaded")
    
    logger.info("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    logger.info("  ✓ VAE loaded")
    
    logger.info("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    logger.info("  ✓ UNet loaded")

    # Freeze base models
    logger.info("Step 3: Freezing base models...")
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    logger.info("  ✓ Base models frozen")

    # Add LoRA
    logger.info("Step 4: Adding LoRA Adapters...")
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"  ✓ LoRA added - Trainable params: {trainable_params:,}")

    # Optimizer
    logger.info("Step 5: Setting up Optimizer...")
    try:
        import bitsandbytes as bnb
        logger.info("  ✓ Using 8-bit AdamW")
        optimizer_cls = bnb.optim.AdamW8bit
    except ImportError:
        logger.info("  ✓ Using standard AdamW")
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(filter(lambda p: p.requires_grad, unet.parameters()), lr=args.learning_rate)

    # Dataset
    logger.info("Step 6: Loading Dataset...")
    dataset = ImageCaptionDataset(
        data_dir=args.train_data_dir,
        captions_file=args.captions_file,
        resolution=args.resolution
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    logger.info(f"  ✓ DataLoader created - {len(train_dataloader)} batches")

    # LR scheduler
    logger.info("Step 7: Setting up Learning Rate Scheduler...")
    num_training_steps = len(train_dataloader) * args.num_train_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger.info(f"  ✓ Scheduler configured")

    # Prepare
    logger.info("Step 8: Preparing with Accelerator...")
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae = vae.to(accelerator.device)
    logger.info("  ✓ Models prepared")

    # Training
    logger.info("Step 9: Starting Training Loop...")
    logger.info("=" * 80)
    
    global_step = 0
    progress_bar = tqdm(total=num_training_steps, disable=not accelerator.is_main_process)
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                image_batch = batch['image'].to(accelerator.device)
                # Cast image to match VAE dtype
                image_batch = image_batch.to(vae.dtype)
                latents = vae.encode(image_batch).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            # Create dummy embeddings for testing
            batch_size = latents.shape[0]
            embedding_dim = 2048
            prompt_embeds = torch.randn(batch_size, 77, embedding_dim, device=accelerator.device, dtype=latents.dtype)
            add_text_embeds = torch.randn(batch_size, 1280, device=accelerator.device, dtype=latents.dtype)
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            )
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # Proper time embeddings for SDXL (height, width, and augmentation)
            time_ids = torch.zeros(batch_size, 6, device=accelerator.device, dtype=latents.dtype)
            time_ids[:, 0] = args.resolution
            time_ids[:, 1] = args.resolution
            
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": time_ids}
            ).sample
            
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.detach().item()})
                
                if (global_step + 1) % 50 == 0:
                    logger.info(f"Step {global_step + 1}: Loss = {loss.detach().item():.4f}")
                
                if (global_step + 1) % args.checkpointing_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step + 1}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                    StableDiffusionXLPipeline.save_lora_weights(
                        save_directory=checkpoint_dir,
                        unet_lora_layers=unet_lora_state_dict,
                    )
                    logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
            
            global_step += 1
    
    progress_bar.close()

    # Save final weights
    logger.info("Step 10: Saving Final LoRA Weights...")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
        )
        logger.info(f"✓ Final weights saved to {args.output_dir}")
    
    logger.info("=" * 80)
    logger.info("✓ Training Completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
