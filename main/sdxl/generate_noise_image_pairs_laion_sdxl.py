from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from main.sdxl.sdxl_text_encoder import SDXLTextEncoder
from accelerate.utils import ProjectConfiguration
from main.utils import SDTextDataset
from transformers import AutoTokenizer
from accelerate.utils import set_seed
from accelerate import Accelerator
from tqdm import tqdm 
import numpy as np 
import argparse 
import torch 
import os 
    
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--folder", type=str, required=True, help="path to folder")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=1250)
    parser.add_argument("--guidance_scale", type=float, default=8)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--latent_channel", type=int, default=4)
    parser.add_argument("--revision", type=str)

    args = parser.parse_args()

    os.makedirs(args.folder, exist_ok=True)

    # initialize accelerator
    accelerator_project_config = ProjectConfiguration(logging_dir=args.folder)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb",
        project_config=accelerator_project_config
    )

    # make sure that different processes don't have the same seed, otherwise they will generate the same images
    set_seed(args.seed + accelerator.process_index)
    print(accelerator.state)

    # use TF32 for faster training on Ampere GPUs
    # disable for older GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None

    text_encoder = SDXLTextEncoder(args, accelerator).to(accelerator.device)

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
    )

    tokenizer_two = AutoTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer_2", revision=args.revision, use_fast=False
    )

    caption_dataset = SDTextDataset(
        args.prompt_path, 
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        is_sdxl=True
    )   

    # split the dataset across gpus 
    # NOTE: current code doesn't handle node failures
    subset_start_index = accelerator.process_index / accelerator.num_processes * len(caption_dataset)
    subset_end_index = (accelerator.process_index + 1) / accelerator.num_processes * len(caption_dataset)

    print(f"Process {accelerator.process_index} has indices {subset_start_index} to {subset_end_index}")

    caption_dataset = torch.utils.data.Subset(
        caption_dataset, 
        list(range(int(subset_start_index), int(subset_end_index)))
    )
    caption_dataloader = torch.utils.data.DataLoader(
        caption_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
    ) # we do shuffle in case we need a randomized subset of the data

    latents_list, images_list, prompt_embeds_list, pooled_prompt_embeds_list = [], [], [], []

    for batch_index, data in tqdm(enumerate(caption_dataloader), disable=not accelerator.is_main_process, total=args.num_batches):
        prompt_embed, pooled_prompt_embed = text_encoder(data)
        uncond_prompt_embed, uncond_pooled_prompt_embed = (
            torch.zeros_like(prompt_embed), torch.zeros_like(pooled_prompt_embed)
        )

        input_latents = torch.randn(
            len(prompt_embed), 
            args.latent_channel, 
            args.latent_resolution, 
            args.latent_resolution,
            device=accelerator.device,
            dtype=torch.float32
        ).half()

        output_images = pipeline(
            prompt_embeds=prompt_embed,
            pooled_prompt_embeds=pooled_prompt_embed,
            negative_prompt_embeds=uncond_prompt_embed,
            negative_pooled_prompt_embeds=uncond_pooled_prompt_embed,
            latents=input_latents,
            output_type="latent",
            guidance_scale=args.guidance_scale
        ) 

        # save as fp16 to save space
        input_latents = input_latents.cpu().half().numpy()
        output_images = output_images.cpu().half().numpy()

        prompt_embeds = prompt_embed.cpu().half().numpy()
        pooled_prompt_embeds = pooled_prompt_embed.cpu().half().numpy()

        latents_list.append(input_latents)
        images_list.append(output_images)
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        if batch_index >= args.num_batches: # early stop
            break

        if batch_index % 250 == 0:
            data_dict = {
                "latents": np.concatenate(latents_list, axis=0),
                "images": np.concatenate(images_list, axis=0),
                "prompt_embeds_list": np.concatenate(prompt_embeds_list, axis=0),
                "pooled_prompt_embeds": np.concatenate(pooled_prompt_embeds_list, axis=0)
            }
            output_path = os.path.join(args.folder, f"BATCH_{batch_index}_noise_image_pairs_{accelerator.process_index:03d}.pt")
            torch.save(
                data_dict, output_path, pickle_protocol=5 
            )

            if os.path.exists(
                os.path.join(args.folder, f"BATCH_{batch_index-250}_noise_image_pairs_{accelerator.process_index:03d}.pt")
            ):
                os.remove(
                    os.path.join(args.folder, f"BATCH_{batch_index-250}_noise_image_pairs_{accelerator.process_index:03d}.pt")
                )
            
    data_dict = {
        "latents": np.concatenate(latents_list, axis=0),
        "images": np.concatenate(images_list, axis=0),
        "prompt_embeds_list": np.concatenate(prompt_embeds_list, axis=0),
        "pooled_prompt_embeds": np.concatenate(pooled_prompt_embeds_list, axis=0)
    }
    output_path = os.path.join(args.folder, f"noise_image_pairs_{accelerator.process_index:03d}.pt")
    torch.save(
        data_dict, output_path, pickle_protocol=5 
    )
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()