#!/bin/bash
# script for training the "lili andrea fiallo class"

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/mnt/sd_models/lilifiallo/"
export OUTPUT_DIR="/mnt/sd_models/"
export CLASS_DIR="/mnt/sd_models/images"

accelerate launch train_dreambooth.py \
	--pretrained_model_name_or_path=$MODEL_NAME \    
	--instance_data_dir=$INSTANCE_DIR \ 
	--class_data_dir=$CLASS_DIR \  
	--output_dir=$OUTPUT_DIR \
	--instance_prompt="lili andrea fiallo" \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --gradient_checkpointing \   
        --use_8bit_adam \
        --enable_xformers_memory_efficient_attention \   
	--set_grads_to_none\
        --learning_rate=2e-6\
        --lr_scheduler="constant"\
        --lr_warmup_steps=0\
        --max_train_steps=800\
        --train_text_encoder
        --with_prior_preservation\
        --prior_loss_weight=1.0\
        --num_class_images=10

