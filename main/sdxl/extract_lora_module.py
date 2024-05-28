from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
import argparse 
import torch 

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--original_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    
    args = parser.parse_args()

    generator = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet"
    ).float()

    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_in",
        "proj_out",
        "ff.net.0.proj",
        "ff.net.2",
        "conv1",
        "conv2",
        "conv_shortcut",
        "downsamplers.0.conv",
        "upsamplers.0.conv",
        "time_emb_proj",
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    generator.add_adapter(lora_config) 

    generator.load_state_dict(torch.load(args.original_model_path))

    if args.fp16:
        generator = generator.half()

    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(generator))
    StableDiffusionXLPipeline.save_lora_weights(args.output_model_path, unet_lora_layers=unet_lora_state_dict)


if __name__ == "__main__":
    main()