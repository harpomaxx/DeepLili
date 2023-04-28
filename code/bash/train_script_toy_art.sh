#!/bin/bash
# script for training the "toy art class"

export MODEL_NAME="CompVis/stable-diffusion-v1-5"
export INSTANCE_DIR="/mnt/sd_models/toyartfull/"
export OUTPUT_DIR="/mnt/sd_models_toyart/"
export CLASS_DIR="/mnt/sd_models/images"


accelerate launch train_dreambooth.py \ 
	--pretrained_model_name_or_path=$MODEL_NAMEi \    
	 --instance_data_dir=$INSTANCE_DIR \   
	 --class_data_dir=$CLASS_DIR \
	 --output_dir=$OUTPUT_DIR \
   	 --instance_prompt="sks style" \
         --resolution=512 \
         --train_batch_size=1 \
         --gradient_accumulation_steps=1 \
         --gradient_checkpointing \
         --use_8bit_adam \
         --enable_xformers_memory_efficient_attention \
         --set_grads_to_none \
         --learning_rate=1e-6 \
         --lr_scheduler="constant" \
         --lr_warmup_steps=0 \
         --max_train_steps=2000 \
         --train_text_encoder \
         --mixed_precision=fp16 
