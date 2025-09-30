

export date_today=9.29
export POST_EXP_NAME=use_flow_matching_libero_4_task
export TENSORBOARD_DIR=logs/${date_today}/${POST_EXP_NAME}




export HF_ENDPOINT=https://hf-mirror.com



current_time=$(date "+%Y-%m-%d_%H-%M-%S")



data_name=libero_4_task_suites_no_noops


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--vla_path pretrained_models/minivla \
--conf_path pretrained_models/minivla/config.json \
--data_root_dir data/modified_libero_rlds \
--dataset_name $data_name \
--run_root_dir openvla-oft/outputs/${date_today} \
--use_l1_regression False \
--use_diffusion False \
--use_flow_matching True \
--use_film False \
--num_images_in_input 1 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivla True \
--merge_lora_during_training True \
--batch_size 32 \
--learning_rate 1e-4 \
--num_steps_before_decay 60000 \
--max_steps 80005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--image_aug True \
--lora_rank 64 \
--use_tensorboard True \
--version 'v1' \
--phase 'stage1' \
--run_id_note v1--minivla--lora64a128--token_64--4_task--$current_time 2>&1 | tee logs/output.log
# > logs/v1--minivla--lora64a128--token_64--4_task--$current_time.log 2>&1 &