# VLA-RFT: Vision-Language-Action Models with Reinforcement Fine-Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![arXiv 2510.00406](https://img.shields.io/badge/arXiv-2510.00406-b31b1b?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2510.00406)

<div id="top" align="center">
<p align="center">
<img src=image/Figure1.png width=90% />
</p>
</div>

<div id="top" align="center">
<p align="center">
<img src=image/Figure2.png width=90% />
</p>
</div>

Vision-Language-Action (VLA) models enable embodied decision-making but rely heavily on imitation learning, leading to compounding errors and poor robustness under distribution shift. Reinforcement learning (RL) can mitigate these issues yet typically demands costly real-world interactions or suffers from sim-to-real gaps. We introduce VLA-RFT, a reinforcement fine-tuning framework that leverages a data-driven world model as a controllable simulator. Trained from real interaction data, the simulator predicts future visual observations conditioned on actions, allowing policy rollouts with dense, trajectory-level rewards derived from goal-achieving references. This design delivers an efficient and action-aligned learning signal, drastically lowering sample requirements. **With fewer than 400 fine-tuning steps, VLA-RFT surpasses strong supervised baselines and achieves greater efficiency than simulator-based RL.** Moreover, it exhibits strong robustness under perturbed conditions, sustaining stable task execution. Our results establish world-model-based RFT as a practical post-training paradigm to enhance the generalization and robustness of VLA models. 

> **🌍 Project page: https://vla-rft.github.io/**<br/>


## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.2+
- uv package manager

### Installation


```bash
# Clone the repository
git clone https://github.com/OpenHelix-Team/VLA-RFT.git
cd VLA-RFT

# Set up the environment
uv venv --seed -p 3.10
source .venv/bin/activate

# Install dependencies
cd train/verl
uv pip install -e ".[gpu]"

# Download the required flash-attn package from: https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0.post1/flash_attn-2.6.0.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install /path/to/flash_attn-2.6.0.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install -e ".[vllm]"

# Install vla-adapter
# Download the required dlimp package from:https://github.com/moojink/dlimp_openvla
uv pip install /path/to/dlimp
uv pip install -r requirements.txt
cd vla-adapter/openvla-oft
uv pip install -e .

# Install LIBERO requirements
cd ../../../../eval/
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
uv pip install -e LIBERO
```


### Basic Usage

#### LIBERO Evaluation Example
```bash
# Run evaluation with LIBERO tasks
cd scripts/libero
bash eval_libero.sh
```

#### Training Example

```bash
# Run training with LIBERO dataset
cd scripts/libero
bash post_train_rlvr.sh
```

## 📊 Supported Tasks & Benchmarks

### LIBERO Benchmark
- **LIBERO-Spatial**: Spatial reasoning tasks
- **LIBERO-Object**: Object manipulation tasks  
- **LIBERO-Goal**: Goal-conditioned tasks
- **LIBERO-10**: 10-task suite

<div id="top" align="center">
<p align="center">
<img src=image/Figure3.png width=90% />
</p>
</div>

## 📈 Performance

With fewer than 400 fine-tuning steps, VLA-RFT surpasses strong supervised baselines and achieves greater efficiency than simulator-based RL.

<div id="top" align="center">
<p align="center">
<img src=image/Table1.png width=90% />
</p>
</div>

<div id="top" align="center">
<p align="center">
<img src=image/Table2.png width=90% />
</p>
</div>

*Please refer to our paper for detailed benchmark results.*

## 📝 TODO

- [x] Init codebase
- [ ] Release pre-trained and rft VLA(policy) weights
- [ ] Release pre-trained World Model weights
- [ ] Support real-world deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use VLA-RFT in your research, please cite:

```bibtex
@article{wang2025vlaadapter,
  author={Wang, Yihao and Ding, Pengxiang and Li, Lingxiao and Cui, Can and Ge, Zirui and Tong, Xinyang and Song, Wenxuan and Zhao, Han and Zhao, Wei and Hou, Pengxu and Huang, Siteng and Tang, Yifan and Wang, Wenhui and Zhang, Ru and Liu, Jianyi and Wang, Donglin},
  title={VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model},
  journal={arXiv preprint arXiv:2509.09372},
  year={2025}
}
```
```bibtex
@misc{li2025vlarftvisionlanguageactionreinforcementfinetuning,
      title={VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators}, 
      author={Hengtao Li and Pengxiang Ding and Runze Suo and Yihao Wang and Zirui Ge and Dongyuan Zang and Kexian Yu and Mingyang Sun and Hongyin Zhang and Donglin Wang and Weihua Su},
      year={2025},
      eprint={2510.00406},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.00406}, 
      }
```


## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:
- [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter): Foundation vision-language-action adapter model
- [VERL](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning framework
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): Lifelong robot learning benchmark
- [RLVR-world](https://github.com/thuml/RLVR-World): Training world model with verified reward

---

**⭐ Star this repository if you find it helpful!**
