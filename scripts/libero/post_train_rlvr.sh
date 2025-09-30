
source .venv/bin/activate

export DATE=$(date +%Y%m%d)
export CKPT_STEPS=150000
export LIBERO_TASK_NAME=10
export POST_EXP_NAME=${LIBERO_TASK_NAME}_fm

mkdir -p logs/libero/rlvr/${DATE}/ckpt_${CKPT_STEPS}_${POST_EXP_NAME}

export HYDRA_FULL_ERROR=1
export TENSORBOARD_DIR=logs/libero/rlvr/${DATE}/ckpt_${CKPT_STEPS}_${POST_EXP_NAME}
export NCCL_P2P_DISABLE=1

export VLLM_ATTENTION_BACKEND=XFORMERS
bash train/verl/examples/grpo_trainer/run_vla_rft.sh 2>&1 | tee logs/libero/output.log