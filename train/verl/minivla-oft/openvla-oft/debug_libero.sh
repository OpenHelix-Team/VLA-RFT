conda init
source activate openvla-oft
export HF_ENDPOINT=https://hf-mirror.com

# cd data
# pip install -e LIBERO
# pip install -r ../openvla-oft/experiments/robot/libero/libero_requirements.txt  # From openvla-oft base dir

# cd ../openvla-oft 
# pip install -e .

cd /dingpengxiang/Pengxiang/minivla-oft/openvla-oft

# 获取当前时间，格式为 "YYYY-MM-DD HH:MM:SS"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# 输出当前时间
echo "当前时间是: $current_time"

data_name=libero_10_no_noops

# 后台运行命令，将输出和错误日志记录到同一个文件
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
--vla_path pretrained_models/minivla \
--conf_path pretrained_models/minivla/config.json \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_l1_regression True \
--use_diffusion False \
--use_flow_matching False \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivla True \
--merge_lora_during_training True \
--batch_size 16 \
--learning_rate 1e-4 \
--num_steps_before_decay 50000 \
--max_steps 40005 \
--save_freq 10000 \
--save_latest_checkpoint_only False \
--image_aug True \
--lora_rank 64 \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--version 'v1' \
--phase 'stage1' \
--run_id_note v1--minivla--lora64a128--token_64--$current_time \