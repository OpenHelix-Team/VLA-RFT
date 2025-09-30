set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_vgpt_trainer \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    model.tokenizer_path=ivideogpt/pretrained_models/checkpoint-tokenizer400000 \
    trainer.total_training_steps=1000000 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=vgpt-pt \
    trainer.experiment_name=vgpt-pt \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@