
LIBERO_TASK_NAME=libero_goal

CKPT_NAME=sft150000

POST_EXP_NAME=rlvr_400

date_today=$(date +%m.%d)



MODEL_DIR=/path/to/your/model/

cd train/verl/minivla-oft/openvla-oft

export HF_ENDPOINT=https://hf-mirror.com


current_time=$(date "+%Y-%m-%d_%H-%M-%S")


# DISTURB_INDEX="[3,8,10]"       # 10Task4

DISTURB_INDEX="[3,10,24,31,38]"     # 10Task5


DISTURB_RANGE=0.025             

TASK_ID=5                       


CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
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
  --save_plot False \
  --ckpt_name ${CKPT_NAME} \
  --plot_path plots/${CKPT_NAME}/${LIBERO_TASK_NAME}_eval_${POST_EXP_NAME}_${date_today} \
  --run_single_task True \
  --single_task_id $TASK_ID \
  --disturb True \
  --disturb_index ${DISTURB_INDEX} \
  --disturb_range ${DISTURB_RANGE} \
  2>&1 | tee logs/libero/pos_disturb/${LIBERO_TASK_NAME}/${TASK_ID}/${CKPT_NAME}_${POST_EXP_NAME}/${DISTURB_INDEX}_${DISTURB_RANGE}_${DISTURB_SEED}_${DISTURB_SEED_RANGE}.log
