
source .venv/bin/activate
export PYTHONPATH="$(pwd)/third_party/LIBERO:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com

LIBERO_TASK_NAME=10                               # e.g., spatial, object, goal, 10
CKPT_NAME=vla_adapter_w_fm_head
POST_EXP_NAME=RFT_400

MODEL_DIR=path/to/your/model/checkpoint           # e.g., checkpoints/libero/RFT/object/20231010_vla_adapter_w_fm_head/checkpoint.pth

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
echo "Current Time : $current_time"

mkdir -p logs/libero/eval/${CKPT_NAME}/${LIBERO_TASK_NAME}

CUDA_VISIBLE_DEVICES=0 python train/verl/vla-adapter/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --use_l1_regression False \
  --use_diffusion False \
  --use_flow_matching True \
  --use_proprio True \
  --use_film False \
  --num_images_in_input 1 \
  --pretrained_checkpoint ${MODEL_DIR} \
  --task_suite_name libero_${LIBERO_TASK_NAME} \
  --save_version v1 \
  --use_minivla True \
  --run_id_note ${POST_EXP_NAME} \
  --local_log_dir logs/libero/eval/${CKPT_NAME}/${LIBERO_TASK_NAME} \