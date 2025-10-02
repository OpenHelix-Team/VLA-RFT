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
# Install dependencies
uv pip install -e third_party/flash-attn-2.6.0.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install -e third_party/dlimp
uv pip install -e third_party/LIBERO
```