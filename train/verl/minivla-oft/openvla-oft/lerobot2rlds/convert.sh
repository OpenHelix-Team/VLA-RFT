# 注意一下

# src-dir是原始lerobot数据集，depth_blocks_0509是总体任务名，yellow_bloc是任务名
# output-dir是存储的rlds的地址
# task-name是后续的任务名


python lerobot2rlds.py \
    --src-dir /ssdwork/Pengxiang/data/depth_blocks_0509/yellow_block \
    --output-dir /ssdwork/Pengxiang/data/realworld \
    --task-name yellow_block