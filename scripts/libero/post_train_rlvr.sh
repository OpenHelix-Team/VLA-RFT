source /202431205128/config/enable_network.sh
cd /202431205128/baseline/COPY/MARVEL
source .venv/bin/activate
cd train/verl

export HYDRA_FULL_ERROR=1
export DATA_TODAY=9.23
export CKPT_STEPS=150000
export LIBERO_TASK_NAME=10
export POST_EXP_NAME=${LIBERO_TASK_NAME}_fm_wacq_seq_9
export TENSORBOARD_DIR=/202431205128/baseline/COPY/MARVEL/logs/libero/rlvr/${DATA_TODAY}/ckpt_${CKPT_STEPS}_${POST_EXP_NAME}
export NCCL_P2P_DISABLE=1

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
export VLLM_ATTENTION_BACKEND=XFORMERS
bash examples/grpo_trainer/run_marvel.sh 2>&1 | tee /202431205128/baseline/COPY/MARVEL/logs/libero/output1.log