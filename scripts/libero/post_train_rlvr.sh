
cd train/verl

export HYDRA_FULL_ERROR=1
export DATE_TODAY=9.23
export CKPT_STEPS=150000
export LIBERO_TASK_NAME=10
export POST_EXP_NAME=${LIBERO_TASK_NAME}_fm_wacq_seq_9
export TENSORBOARD_DIR=logs/libero/rlvr/${DATE_TODAY}/ckpt_${CKPT_STEPS}_${POST_EXP_NAME}


# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
export VLLM_ATTENTION_BACKEND=XFORMERS
bash examples/grpo_trainer/run_marvel.sh 2>&1 | tee logs/libero/output1.log