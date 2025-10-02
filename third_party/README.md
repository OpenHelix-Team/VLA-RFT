### If your network is restricted, please download the dependencies manually from the following links:


```bash
# Download the LIBERO submodule
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO

# Download flash-attention wheel
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0.post1/flash_attn-2.6.0.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -P third_party/

# Download dlimp package
git clone https://github.com/moojink/dlimp.git third_party/dlimp
```

### Then, follow these steps to install the dependencies:

```bash
# 1) Set up the environment
uv venv --seed -p 3.10
source .venv/bin/activate

# 2) Install dependencies
uv pip install -e train/verl/".[gpu]"
uv pip install -e third_party/flash-attn-2.6.0.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install -e train/verl/".[vllm]"
uv pip install -r train/verl/requirements.txt

# 3) Install vla-adapter
uv pip install -e third_party/dlimp
uv pip install -e train/verl/vla-adapter/openvla-oft

# 4) Install LIBERO requirements
uv pip install -e third_party/LIBERO
```