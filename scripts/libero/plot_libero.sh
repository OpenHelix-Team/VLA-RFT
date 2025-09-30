cd /202431205128/baseline/COPY/MARVEL
source .venv/bin/activate
export DATA_TODAY=9.11
DIR_SAVE_NAME='SPATIAL_TASK_4_1'

# 获取当前时间，格式为 "YYYY-MM-DD HH:MM:SS"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# 输出当前时间
echo "当前时间是: $current_time"

python plots/plot_libero.py \
    /202431205128/baseline/COPY/MARVEL/plots/BASE/libero_spatial_eval_sft90000_9.11/action_data/task_4_BASE.json \
    /202431205128/baseline/COPY/MARVEL/plots/RLVR/libero_spatial_eval_sft90000rl400_9.11/action_data/task_4_RLVR.json \
    --output_dir /202431205128/baseline/COPY/MARVEL/plots/${DATA_TODAY}/${DIR_SAVE_NAME} \
    --traj_indices "1-3,6-7,9-12,15-17,21-23,27-29,32-34,39-41,44-45,46,48-49,8" \
    # --success_only \
