from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler, AutoencoderTiny
from main.sdxl.sdxl_text_encoder import SDXLTextEncoder
from main.utils import get_x0_from_noise
from transformers import AutoTokenizer
from accelerate import Accelerator
import gradio as gr    
import numpy as np
import argparse 
import torch
import time 
import PIL
    
SAFETY_CHECKER = False

class ModelWrapper:
    def __init__(self, args, accelerator):
        super().__init__()
        # disable all gradient calculations
        torch.set_grad_enabled(False)
        
        if args.precision == "bfloat16":
            self.DTYPE = torch.bfloat16
        elif args.precision == "float16":
            self.DTYPE = torch.float16
        else:
            self.DTYPE = torch.float32
        self.device = accelerator.device

        self.tokenizer_one = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )

        self.text_encoder = SDXLTextEncoder(args, accelerator).to(dtype=self.DTYPE)

        # Initialize AutoEncoder with specified model and dtype
        if args.use_tiny_vae:
            self.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", 
                torch_dtype=self.DTYPE
            ).to(self.device)
        else:
            self.vae = AutoencoderKL.from_pretrained(
                args.model_id, 
                subfolder="vae"
            ).to(self.device).float()

        # Initialize Generator
        self.model = self.create_generator(args).to(dtype=self.DTYPE).to(self.device)

        self.accelerator = accelerator
        self.image_resolution = args.image_resolution
        self.latent_resolution = args.latent_resolution
        self.num_train_timesteps = args.num_train_timesteps

        self.base_add_time_ids = self.build_condition_input()
        self.conditioning_timestep = args.conditioning_timestep 

        self.scheduler = DDIMScheduler.from_pretrained(
            args.model_id,
            subfolder="scheduler"
        )
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # sampling parameters 
        self.num_step = args.num_step 
        self.conditioning_timestep = args.conditioning_timestep 

        # safety checker 
        if SAFETY_CHECKER:
            # adopted from https://huggingface.co/spaces/ByteDance/SDXL-Lightning/raw/main/app.py
            from demo.safety_checker import StableDiffusionSafetyChecker
            from transformers import CLIPFeatureExtractor

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(self.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32", 
            )

    def check_nsfw_images(self, images):
        safety_checker_input = self.feature_extractor(images, return_tensors="pt") # .to(self.dviece)
        has_nsfw_concepts = self.safety_checker(
            clip_input=safety_checker_input.pixel_values.to(self.device),
            images=images
        )
        return has_nsfw_concepts

    def create_generator(self, args):
        generator = UNet2DConditionModel.from_pretrained(
            args.model_id,
            subfolder="unet"
        ).to(self.DTYPE)

        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        print(generator.load_state_dict(state_dict, strict=True))
        generator.requires_grad_(False)
        return generator 

    def build_condition_input(self):
        original_size = (self.image_resolution, self.image_resolution)
        target_size = (self.image_resolution, self.image_resolution)
        crop_top_left = (0, 0)

        add_time_ids = list(original_size + crop_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=self.device, dtype=self.DTYPE)
        return add_time_ids

    def _encode_prompt(self, prompt):
        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_two = self.tokenizer_two(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_dict = {
            'text_input_ids_one': text_input_ids_one.unsqueeze(0).to(self.device),
            'text_input_ids_two': text_input_ids_two.unsqueeze(0).to(self.device)
        }
        return prompt_dict 

    @staticmethod
    def _get_time():
        torch.cuda.synchronize()
        return time.time()

    def sample(
        self, noise, unet_added_conditions, prompt_embed
    ):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        if self.num_step == 1:
            all_timesteps = [self.conditioning_timestep]
            step_interval = 0 
        elif self.num_step == 4:
            all_timesteps = [999, 749, 499, 249]
            step_interval = 250 
        else:
            raise NotImplementedError()
        
        DTYPE = prompt_embed.dtype
        
        for constant in all_timesteps:
            current_timesteps = torch.ones(len(prompt_embed), device=self.device, dtype=torch.long)  *constant
            eval_images = self.model(
                noise, current_timesteps, prompt_embed, added_cond_kwargs=unet_added_conditions
            ).sample

            eval_images = get_x0_from_noise(
                noise, eval_images, alphas_cumprod, current_timesteps
            ).to(self.DTYPE)

            next_timestep = current_timesteps - step_interval 
            noise = self.scheduler.add_noise(
                eval_images, torch.randn_like(eval_images), next_timestep
            ).to(DTYPE)  

        eval_images = self.vae.decode(eval_images / self.vae.config.scaling_factor, return_dict=False)[0]
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return eval_images 


    @torch.no_grad()
    def inference(
        self,
        prompt: str,
        seed: int,
        num_images: int=1,
    ):
        print("Running model inference...")

        if seed == -1:
            seed = np.random.randint(0, 1000000)

        generator = torch.manual_seed(seed)

        add_time_ids = self.base_add_time_ids.repeat(num_images, 1)

        noise = torch.randn(
            num_images, 4, self.latent_resolution, self.latent_resolution, 
            generator=generator
        ).to(device=self.device, dtype=self.DTYPE) 

        prompt_inputs = self._encode_prompt(prompt)
        
        start_time = self._get_time()

        prompt_embeds, pooled_prompt_embeds = self.text_encoder(prompt_inputs)

        batch_prompt_embeds, batch_pooled_prompt_embeds = (
            prompt_embeds.repeat(num_images, 1, 1),
            pooled_prompt_embeds.repeat(num_images, 1, 1)
        )

        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": batch_pooled_prompt_embeds.squeeze(1)
        }

        eval_images = self.sample(
            noise=noise,
            unet_added_conditions=unet_added_conditions,
            prompt_embed=batch_prompt_embeds
        )

        end_time = self._get_time()

        output_image_list = [] 
        for image in eval_images:
            output_image_list.append(PIL.Image.fromarray(image.cpu().numpy()))

        if SAFETY_CHECKER:
            has_nsfw_concepts = self.check_nsfw_images(output_image_list)
            if any(has_nsfw_concepts):
                return [PIL.Image.new("RGB", (512, 512))], "NSFW concepts detected. Please try a different prompt."

        return (
            output_image_list,
            f"run successfully in {(end_time-start_time):.2f} seconds"
        )


def create_demo():
    TITLE = "# DMD2-SDXL Demo"
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--image_resolution", type=int, default=1024)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--use_tiny_vae", action="store_true")
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--num_step", type=int, default=4, choices=[1, 4])
    parser.add_argument("--revision", type=str)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 

    accelerator = Accelerator()

    model = ModelWrapper(args, accelerator)

    with gr.Blocks() as demo:
        gr.Markdown(TITLE)
        with gr.Row():
            with gr.Column():
                prompt = gr.Text(
                    value="children",
                    label="Prompt",
                    placeholder='e.g. children'
                )
                run_button = gr.Button("Run")
                with gr.Accordion(label="Advanced options", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=-1,
                        maximum=1000000,
                        step=1,
                        value=0,
                        info="If set to -1, a different seed will be used each time.",
                    )
                    num_images = gr.Slider(
                        label="Number of generated images",
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=1,
                    )
            with gr.Column():
                result = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery", height=1024)

                error_message = gr.Text(label="Job Status")

        inputs = [
            prompt,
            seed,
            num_images
        ]
        run_button.click(
            fn=model.inference, inputs=inputs, outputs=[result, error_message]
        )
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue(api_open=True)
    demo.launch(
        server_name="0.0.0.0",
        show_error=True,
        share=True
    )
