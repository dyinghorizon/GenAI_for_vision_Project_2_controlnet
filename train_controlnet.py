
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = get_logger(__name__)

class ControlNetDataset(Dataset):
    def __init__(self, data_dir, tokenizer, resolution=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.metadata = pd.read_csv(self.data_dir / "metadata.csv")
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.control_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        image = Image.open(self.data_dir / row['image']).convert('RGB')
        control = Image.open(self.data_dir / row['control']).convert('RGB')
        prompt = row['prompt']
        
        image = self.image_transforms(image)
        control = self.control_transforms(control)
        
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            'pixel_values': image,
            'conditioning_pixel_values': control,
            'input_ids': input_ids
        }

def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    conditioning_pixel_values = torch.stack([example['conditioning_pixel_values'] for example in examples])
    input_ids = torch.stack([example['input_ids'] for example in examples])
    
    return {
        'pixel_values': pixel_values,
        'conditioning_pixel_values': conditioning_pixel_values,
        'input_ids': input_ids
    }

def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    controlnet_id = "lllyasviel/control_v11f1p_sd15_depth"
    train_data_dir = "/workspace/data/dataset/train"
    val_data_dir = "/workspace/data/dataset/val"
    output_dir = "/workspace/controlnet-interior"
    
    resolution = 512
    train_batch_size = 4
    gradient_accumulation_steps = 2
    learning_rate = 1e-5
    max_train_steps = 25000
    checkpointing_steps = 2500
    validation_steps = 1000
    mixed_precision = "fp16"
    seed = 42
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs")
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_id)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    controlnet.enable_gradient_checkpointing()
    
    if hasattr(unet, 'enable_xformers_memory_efficient_attention'):
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=learning_rate)
    
    train_dataset = ControlNetDataset(train_data_dir, tokenizer, resolution)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=4
    )
    
    val_dataset = ControlNetDataset(val_data_dir, tokenizer, resolution)
    
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_train_steps
    )
    
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    train_loss_log = []
    
    while global_step < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(controlnet):
                latents = vae.encode(batch['pixel_values'].to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]
                
                controlnet_image = batch['conditioning_pixel_values']
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss_log.append(loss.detach().item())
                
                if global_step % 100 == 0:
                    avg_loss = np.mean(train_loss_log[-100:])
                    progress_bar.set_postfix(loss=avg_loss)
                
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.unwrap_model(controlnet).save_pretrained(save_path)
                        print(f"Saved checkpoint to {save_path}")
                
                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        print(f"Step {global_step}: Running validation...")
                        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                            model_id,
                            controlnet=accelerator.unwrap_model(controlnet),
                            torch_dtype=torch.float16,
                            safety_checker=None,
                        )
                        pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        
                        val_indices = np.random.choice(len(val_dataset), 4, replace=False)
                        val_images = []
                        
                        for idx in val_indices:
                            val_sample = val_dataset[idx]
                            control_image = transforms.ToPILImage()(val_sample['conditioning_pixel_values'])
                            prompt = val_dataset.metadata.iloc[idx]['prompt']
                            
                            with torch.autocast("cuda"):
                                output = pipeline(
                                    prompt=prompt,
                                    image=control_image,
                                    num_inference_steps=20,
                                    guidance_scale=7.5,
                                ).images[0]
                            
                            val_images.append((control_image, output, prompt))
                        
                        fig, axes = plt.subplots(4, 2, figsize=(10, 20))
                        for i, (control, output, prompt) in enumerate(val_images):
                            axes[i, 0].imshow(control)
                            axes[i, 0].set_title("Control")
                            axes[i, 0].axis('off')
                            axes[i, 1].imshow(output)
                            axes[i, 1].set_title(f"Generated")
                            axes[i, 1].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"validation_{global_step}.png"))
                        plt.close()
                        
                        del pipeline
                        torch.cuda.empty_cache()
                
                if global_step % 500 == 0:
                    if accelerator.is_main_process:
                        with open(os.path.join(output_dir, "training_loss.json"), 'w') as f:
                            json.dump(train_loss_log, f)
            
            if global_step >= max_train_steps:
                break
    
    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(output_dir, "controlnet-final"))
        
        with open(os.path.join(output_dir, "training_loss.json"), 'w') as f:
            json.dump(train_loss_log, f)
        
        print("Training complete!")

if __name__ == "__main__":
    main()
