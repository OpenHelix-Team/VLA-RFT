cd /202431205128/baseline/MARVEL
source .venv/bin/activate
cd scripts/libero
export NCCL_P2P_DISABLE=1
python oxe_data_converter.py \
    --dataset_name libero_spatial_no_noops \
    --input_path /202431205128/data/modified_libero_rlds \
    --output_path /202431205128/data/data_256/libero \
    --begin_num_episodes 1261
