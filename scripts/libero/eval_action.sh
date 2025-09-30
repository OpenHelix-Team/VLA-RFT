source /202431205128/config/enable_network.sh

LIBERO_TASK_NAME=libero_goal

CKPT_NAME=sft550000

CKPT_STEP=0

POST_EXP_NAME=rlvr_${CKPT_STEP}

DATA_TODAY=$(date +%m.%d)
export TENSORBOARD_DIR=/202431205128/baseline/minivla-oft/logs/${DATA_TODAY}/${POST_EXP_NAME}
export HUGGINGFACE_HUB_TOKEN=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
export PYTHONPATH="/202431205128/baseline/COPY/MARVEL/eval/LIBERO:$PYTHONPATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless          # 常见的无窗口平台
export MUJOCO_EGL_DEVICE_ID=0            # 指定第0块GPU，如有多卡可改
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NCCL_P2P_DISABLE=1

# MODEL_DIR=/202431205128/baseline/COPY/MARVEL/checkpoints/libero/rlvr/150000base
# MODEL_DIR=/202431205128/baseline/COPY/MARVEL/checkpoints/libero/rlvr/long/sft150000rl/ckpt_150000_fm_wacq_gt_ac_10/global_step_400/actor
# MODEL_DIR=/202431205128/baseline/COPY/MARVEL/checkpoints/libero/rlvr/ckpt_150000/10_fm_wacq_seq_9_9.19/global_step_${CKPT_STEP}/actor
MODEL_DIR=/202431205128/baseline/minivla-oft/openvla-oft/outputs/8.29/minivla+libero_4_task_suites_no_noops+b40+lr-0.0002+lora-r64+dropout-0.0--image_aug--v1--minivla--lora64a128--token_64--4_task--2025-09-27_14-40-51--55000_chkpt
# MODEL_DIR=/202431205128/baseline/COPY/MARVEL/checkpoints/libero/rlvr/ckpt_150000/goal_fm_wacq_seq_9_9.19/global_step_420/actor

cd /202431205128/baseline/COPY/MARVEL
source .venv/bin/activate
cd train/verl/minivla-oft/openvla-oft

export HF_ENDPOINT=https://hf-mirror.com

# 获取当前时间，格式为 "YYYY-MM-DD HH:MM:SS"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# 输出当前时间
echo "当前时间是: $current_time"

TASK_ID=7                       # 指定任务ID
echo "当前任务是: $LIBERO_TASK_NAME"
# echo "当前任务ID是: $TASK_ID"
# echo "当前扰动幅度是: $DISTURB_RANGE, 扰动state的维度是: $DISTURB_INDEX, 扰动随机种子是: $DISTURB_SEED, 扰动随机种子范围是: $DISTURB_SEED_RANGE"
echo "当前模型是: $CKPT_NAME-$POST_EXP_NAME"

# mkdir -p /202431205128/baseline/COPY/MARVEL/logs/libero/pos_disturb/${LIBERO_TASK_NAME}/${TASK_ID}/${CKPT_NAME}_${POST_EXP_NAME}
mkdir -p /202431205128/baseline/COPY/MARVEL/logs/libero/eval/${LIBERO_TASK_NAME}/${TASK_ID}


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
  --ckpt_name ${CKPT_NAME}_${POST_EXP_NAME} \
  --plot_path /202431205128/baseline/COPY/MARVEL/plots/${LIBERO_TASK_NAME}/${TASK_ID}/${CKPT_NAME}_${POST_EXP_NAME} \
  --run_single_task False \
  --single_task_id $TASK_ID \
  2>&1 | tee /202431205128/baseline/COPY/MARVEL/logs/libero/eval/${LIBERO_TASK_NAME}/${TASK_ID}/${CKPT_NAME}_${POST_EXP_NAME}.log
  # 2>&1 | tee /202431205128/baseline/COPY/MARVEL/logs/libero/pos_disturb/${LIBERO_TASK_NAME}/${TASK_ID}/${CKPT_NAME}_${POST_EXP_NAME}/${DISTURB_INDEX}_${DISTURB_RANGE}_${DISTURB_SEED}_${DISTURB_SEED_RANGE}.log
  # 2>&1 | tee /202431205128/baseline/COPY/MARVEL/logs/libero/eval/${CKPT_NAME}/${LIBERO_TASK_NAME}/eval_${POST_EXP_NAME}_eval_${DATA_TODAY}.log
  # 2>&1 | tee /202431205128/baseline/COPY/MARVEL/logs/libero/pos_disturb/${CKPT_NAME}_${POST_EXP_NAME}/${LIBERO_TASK_NAME}_${TASK_ID}/${DISTURB_INDEX}_${DISTURB_RANGE}_${DISTURB_SEED}_${DISTURB_SEED_RANGE}.log
  # 2>&1 | tee /202431205128/baseline/COPY/MARVEL/plots/logs/${CKPT_NAME}/${LIBERO_TASK_NAME}_eval_${POST_EXP_NAME}_${DATA_TODAY}.log  
  # --existing_data_path /202431205128/baseline/COPY/MARVEL/plots/BASE/libero_spatial_eval_sft30000_9.11/action_data/task_5_BASE.json \
  # --pretrained_checkpoint /202431205128/baseline/COPY/MARVEL/checkpoints/libero/rlvr/9.7/ckpt_150000_fm_wacq_gt_ac_object/global_step_420/actor \

