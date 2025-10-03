
source .venv/bin/activate

export DATE=$(date +%Y%m%d)
export LIBERO_TASK_NAME=object
export POST_EXP_NAME=vla_adapter_w_fm_head

mkdir -p logs/libero/RFT/${DATE}

export TENSORBOARD_DIR=logs/libero/RFT/${DATE}/${LIBERO_TASK_NAME}_${POST_EXP_NAME}
export NCCL_P2P_DISABLE=1

export VLLM_ATTENTION_BACKEND=XFORMERS
bash train/verl/examples/grpo_trainer/run_vla_rft.sh 2>&1 | tee logs/libero/RFT/${DATE}/output.log