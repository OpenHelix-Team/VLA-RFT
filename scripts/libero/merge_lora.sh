source /202431205128/config/enable_network.sh
export NCCL_P2P_DISABLE=1
export HUGGINGFACE_HUB_TOKEN=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

cd /202431205128/baseline/COPY/MARVEL
source .venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com



# 获取当前时间，格式为 "YYYY-MM-DD HH:MM:SS"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# 输出当前时间
echo "当前时间是: $current_time"

python train/verl/minivla-oft/openvla-oft/vla-scripts/merge_lora_weights_and_save.py \
    --lora_finetuned_checkpoint_dir /202431205128/baseline/COPY/MARVEL/checkpoints/libero/rlvr/8.22/ckpt_150000_test_w_mse_metrics/global_step_20/actor