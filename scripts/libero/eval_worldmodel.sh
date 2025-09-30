cd /202431205128/baseline/MARVEL
source .venv/bin/activate
cd vid_wm/ivideogpt
export NCCL_P2P_DISABLE=1
bash scripts/eval_multi_step_prediction.sh 2>&1 | tee /202431205128/baseline/MARVEL/logs/libero/eval/output.log