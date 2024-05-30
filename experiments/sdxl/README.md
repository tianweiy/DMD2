## Getting Started with DMD2 on SDXL

### Model Zoo

| Config Name | FID | Link | Iters | Hours |
| ----------- | --- | ---- | ----- | ----- |
| [sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch](./sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh) | 19.32 | [link](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_checkpoint_model_019000) | 19k | 57 |
| [sdxl_cond999_8node_lr5e-5_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_lora](./sdxl_cond999_8node_lr5e-5_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_lora.sh) | 19.68 | [link](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdxl/sdxl_cond999_8node_lr5e-5_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_lora_checkpoint_model_016000) | 16k | 63 |
| [sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode](./sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode.sh) | 19.01 | [link](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdxl/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode_checkpoint_model_024000) | 24k | 57 |


For inference with our models, you only need to download the pytorch_model.bin file from the provided link. For fine-tuning, you will need to download the entire folder.
You can use the following script for that:

```bash 
export CHECKPOINT_NAME="sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_checkpoint_model_019000"  # note that the sdxl/ is necessary
export OUTPUT_PATH="path/to/your/output/folder"

bash scripts/download_hf_checkpoint.sh $CHECKPOINT_NAME $OUTPUT_PATH
```


### Download Base Diffusion Models and Training Data
```bash
export CHECKPOINT_PATH="" # change this to your own checkpoint folder (this should be a central directory shared across nodes)
export WANDB_ENTITY="" # change this to your own wandb entity
export WANDB_PROJECT="" # change this to your own wandb project
export MASTER_IP=""  # change this to your own master ip

# Not sure why but we found the following line necessary to work with the accelerate package in our system. 
# Change YOUR_MASTER_IP/YOUR_MASTER_NODE_NAME to the correct value 
echo "YOUR_MASTER_IP 	YOUR_MASTER_NODE_NAME" | sudo tee -a /etc/hosts

# create a fsdp configs for accelerate launch. change the EXP_NAME to your own experiment name 
python main/sdxl/create_sdxl_fsdp_configs.py --folder fsdp_configs/EXP_NAME  --master_ip $MASTER_IP --num_machines 8  --sharding_strategy 4
mkdir $CHECKPOINT_PATH

bash scripts/download_sdxl.sh $CHECKPOINT_PATH
```

You can also add these few export to the bashrc file so that you don't need to run them every time you open a new terminal.

### 4-step Sample Training/Testing Commands 

```bash
# start a training with 64 gpu. we need to run this script on all 8 nodes. Please change the EXP_NAME and NODE_RANK_ID accordingly.  
bash experiments/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh $CHECKPOINT_PATH $WANDB_ENTITY $WANDB_PROJECT fsdp_configs/EXP_NAME NODE_RANK_ID 

# on some other machine, start a testing process that continually reads from the checkpoint folder and evaluate the FID 
# Change TIMESTAMP_TBD to the real one
python main/sdxl/test_folder_sdxl.py \
    --folder $CHECKPOINT_PATH/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch/TIMESTAMP_TBD/ \
    --conditioning_timestep 999 --num_step 4 --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT --num_train_timesteps 1000 \
    --seed 10 --eval_res 512 --ref_dir $CHECKPOINT_PATH/coco10k/subset \
    --anno_path  $CHECKPOINT_PATH/coco10k/all_prompts.pkl \
    --total_eval_samples 10000 --clip_score \
    --wandb_name test_sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch
```

### 1-step Sample Training/Testing Commands [Work In Progress]

For 1-step model, we need an extra regression loss pretraining. 

First, download the 10K noise-image pairs

```bash
bash scripts/download_sdxl_ode_pair_10k_lmdb.sh $CHECKPOINT_PATH
```

These pairs can be generated using [generate_noise_image_pairs_laion_sdxl.py](../../main/sdxl/generate_noise_image_pairs_laion_sdxl.py)

Second, Pretrain the model with regression loss 

```bash 
bash experiments/sdxl/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399.sh $CHECKPOINT_PATH $WANDB_ENTITY $WANDB_PROJECT $MASTER_IP
```

Alternatively, you can skip the previous two steps and directly download the regression loss pretrained checkpoint 

```bash
bash scripts/download_sdxl_1step_ode_pairs_ckpt.sh $CHECKPOINT_PATH
```

Start the real training 

```bash
# start a training with 64 gpu. we need to run this script on all 8 nodes. Please change the EXP_NAME and NODE_RANK_ID accordingly.  
bash experiments/sdxl/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode.sh $CHECKPOINT_PATH $WANDB_ENTITY $WANDB_PROJECT fsdp_configs/EXP_NAME NODE_RANK_ID 

# on some other machine, start a testing process that continually reads from the checkpoint folder and evaluate the FID 
# Change TIMESTAMP_TBD to the real one
python main/sdxl/test_folder_sdxl.py \
    --folder $CHECKPOINT_PATH/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode/TIMESTAMP_TBD/ \
    --conditioning_timestep 399 --num_step 1 --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT --num_train_timesteps 1000 \
    --seed 10 --eval_res 512 --ref_dir $CHECKPOINT_PATH/coco10k/subset \
    --anno_path  $CHECKPOINT_PATH/coco10k/all_prompts.pkl \
    --total_eval_samples 10000 --clip_score \
    --wandb_name test_sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode
```

Please refer to [train_sd.py](../../main/train_sd.py) for various training options. Notably, if the `--delete_ckpts` flag is set to `True`, all checkpoints except the latest one will be deleted during training. Additionally, you can use the `--cache_dir` flag to specify a location with larger storage capacity. The number of checkpoints stored in `cache_dir` is controlled by the `max_checkpoint` argument.

For LORA training, add the `--generator_lora` flag to the training command. The final checkpoint can be converted to a LORA model using the [extract_lora_module.py](../../main/sdxl/extract_lora_module.py) script.