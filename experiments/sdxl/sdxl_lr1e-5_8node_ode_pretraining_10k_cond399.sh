export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3
export MASTER_IP=$4

torchrun --nnodes 8 --nproc_per_node=8 --rdzv_id=2345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP main/train_sd_ode.py \
    --generator_lr 1e-5  \
    --train_iters 100000000 \
    --output_path  $CHECKPOINT_PATH/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399 \
    --grid_size 1 \
    --log_iters 1000 \
    --resolution 1024 \
    --seed 10 \
    --max_grad_norm 10.0 \
    --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --wandb_iters 250 \
    --wandb_entity tyin \
    --wandb_name "sdxl_lr1e-5_8node_ode_pretraining_10k_cond399"  \
    --sdxl \
    --num_ode_pairs 10000 \
    --ode_pair_path $CHECKPOINT_PATH/laion6.25_pair_generation_sdxl_guidance6_full_lmdb/ \
    --ode_batch_size 4  \
    --conditioning_timestep 399 \
    --tiny_vae \
    --use_fp16 
