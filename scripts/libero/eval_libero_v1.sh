
LIBERO_TASK_NAME=libero_goal

CKPT_NAME=sft150000

POST_EXP_NAME=rlvr_0

date_today=$(date +%m.%d)



MODEL_DIR=checkpoints/libero/rlvr/150000base


export HF_ENDPOINT=https://hf-mirror.com


current_time=$(date "+%Y-%m-%d_%H-%M-%S")

mkdir -p logs/libero/${LIBERO_TASK_NAME}           
           

CUDA_VISIBLE_DEVICES=0 python train/verl/minivla-oft/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --use_l1_regression False \
  --use_diffusion False \
  --use_flow_matching True \
  --use_proprio True \
  --use_film False \
  --num_images_in_input 1 \
  --pretrained_checkpoint ${MODEL_DIR} \
  --task_suite_name ${LIBERO_TASK_NAME} \
  --save_version v1 \
  --use_minivla True \
  --num_trials_per_task 50 \
  2>&1 | tee logs/libero/${LIBERO_TASK_NAME}/${CKPT_NAME}_${POST_EXP_NAME}.log
