from peft import LoraConfig, get_peft_model_state_dict
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
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

    unet_lora_state_dict = get_peft_model_state_dict(generator)

    new_state_dict = {} 

    for k, v in unet_lora_state_dict.items():

        if "lora_A" in k:
            k = k.replace("lora_A", "lora_down")
        elif "lora_B" in k:
            k = k.replace("lora_B", "lora_up")

        k = "lora_unet_" + "_".join(k.split(".")[:-2]) + "." + ".".join(k.split(".")[-2:])

        new_state_dict[k] = v 

        alpha_key = k[:k.find(".")]+".alpha"

        new_state_dict[alpha_key] = torch.tensor(args.lora_alpha, dtype=torch.float16 if args.fp16 else torch.float32)
        
    save_file(new_state_dict, args.output_model_path)

if __name__ == "__main__":
    main()