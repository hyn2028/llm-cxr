#!/bin/bash
export timestamp=`date +%Y-%m-%d_%H-%M-%S`
export model_name='llmcxr'
export checkpoint_dir_name="${model_name}__${timestamp}"
export deepspeed_config=`pwd`/config/ds_z3_bf16_config.json
export local_training_root='./'
export local_output_dir="${local_training_root}/${checkpoint_dir_name}"
export dbfs_output_dir=''
export tensorboard_display_dir="${local_output_dir}/runs"
export input_model="PATH_TO_STAGE1_CHECKPOINT"

deepspeed --num_gpus=8 \
     --module training.trainer \
     --input-model $input_model \
     --deepspeed $deepspeed_config \
     --epochs 5 \
     --local-output-dir $local_output_dir \
     --dbfs-output-dir "" \
     --per-device-train-batch-size 2 \
     --per-device-eval-batch-size 2 \
     --logging-steps 50 \
     --save-total-limit 5 \
     --eval-steps 1500 \
     --warmup-steps 50 \
     --test-size 200 \
     --lr 5e-6 \
     --stage 2
