# VLA-RFT: Vision-Language-Action Models with Reinforcement Fine-Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

VLA-RFT is a comprehensive framework for training and fine-tuning Vision-Language-Action (VLA) models using reinforcement learning techniques. This repository combines state-of-the-art VLA models with advanced RL training methods to achieve superior performance in robotic manipulation tasks.

## 🔥 Key Features

- **VLA-Adapter Architecture**: Built on top of VLA-Adapter with efficient adapter-based fine-tuning
- **Reinforcement Learning Integration**: Leverages VERL (Volcano Engine Reinforcement Learning) framework for efficient RL training
- **Multi-Modal Support**: Handles vision, language, and action modalities seamlessly
- **Flow Matching Training**: Specialized support for flow matching approaches for smooth action generation
- **Comprehensive Evaluation**: Includes evaluation pipelines for LIBERO benchmark tasks
- **Production Ready**: Optimized for both research and deployment scenarios

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU 
- uv package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OpenHelix-Team/VLA-RFT.git
   cd VLA-RFT
   ```

2. **Set up the environment:**
   ```bash
   uv venv --seed -p 3.10
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   install verl with gpu support:
   ```bash
   cd train/verl
   uv pip install -e ".[gpu]"
   ```
   Download the required flash-attn package from:

   https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0.post1/flash_attn-2.6.0.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   
   And install it via pip:
   ```bash
   uv pip install /path/to/flash_attn-2.6.0.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   uv pip install -e ".[vllm]"
   ```
   install minivla-oft and LIBERO requirements:

   Download the required dlimp package from:
   
   https://github.com/moojink/dlimp_openvla
   
   And install it via pip:

   ```bash
   uv pip install /path/to/dlimp
   uv pip install -r requirements.txt
   cd minivla-oft/openvla-oft
   uv pip install -e .
   cd ../../../../eval/
   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
   uv pip install -e LIBERO
   ```
   If anything goes wrong, please refer to the [detailed installation guide](train/verl/minivla-oft/openvla-oft/LIBERO.md).


### Basic Usage

#### LIBERO Evaluation Example
```bash
# Run evaluation with LIBERO tasks
cd scripts/libero
bash eval_libero_v1.sh
```

#### Training Example

```bash
# Run training with LIBERO dataset
cd scripts/libero
bash post_train_rlvr.sh
```

## 📁 Project Structure

```
VLA-RFT/
├── train/                      # Training framework and models
│   └── verl/                   # VERL reinforcement learning framework
│       ├── minivla-oft/        # VLA-Adapter implementation
│       ├── examples/           # Training examples and configurations
│       ├── verl/               # Core VERL implementation
│       └── docs/               # Documentation
├── eval/                       # Evaluation scripts and benchmarks
│   └── LIBERO/                 # LIBERO benchmark evaluation
├── scripts/                    # Utility scripts
│   └── libero/                 # LIBERO-specific scripts
└── README.md                   # This file
```

## 🛠️ Model Architecture

VLA-RFT incorporates several advanced techniques:

- **Vision-Language-Action Fusion**: Multi-modal transformer architecture that processes visual observations, language instructions, and proprioceptive feedback
- **VLA-Adapter**: Efficient adapter-based fine-tuning approach for large vision-language-action models
- **Flow Matching**: Advanced flow-based action generation for smooth, realistic action sequences
- **Reinforcement Learning**: VERL-based RL training for policy optimization

## 📊 Supported Tasks & Benchmarks

### LIBERO Benchmark
- **LIBERO-Spatial**: Spatial reasoning tasks
- **LIBERO-Object**: Object manipulation tasks  
- **LIBERO-Goal**: Goal-conditioned tasks
- **LIBERO-10**: 10-task suite
- **LIBERO-90**: 90-task comprehensive suite

### Training Configurations
- Flow Matching (supported method for smooth action generation)

## 🔧 Configuration Options

The framework supports extensive configuration through YAML files and command-line arguments:

```python
# Key configuration parameters
use_flow_matching = True       # Use flow matching for action generation
num_images_in_input = 1        # Number of input camera views
use_proprio = True             # Include proprioceptive feedback
task_suite_name = "libero_spatial"  # LIBERO task suite to evaluate
```

## 📈 Performance

VLA-RFT achieves state-of-the-art performance on various robotic manipulation benchmarks:

- **LIBERO-Spatial**: XX% success rate
- **LIBERO-Object**: XX% success rate  
- **LIBERO-Goal**: XX% success rate

*Please refer to our paper for detailed benchmark results.*


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use VLA-RFT in your research, please cite:

```bibtex
@article{vla-rft2025,
  title={VLA-RFT: Vision-Language-Action Models with Reinforcement Fine-Tuning},
  author={[Your Name]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 🔗 Related Work

- [VLA-Adapter](https://github.com/vla-adapter/vla-adapter): Foundation vision-language-action adapter model
- [VERL](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning framework
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): Lifelong robot learning benchmark

## 📝 TODO

- [x] Init codebase
- [ ] Release pre-trained code
- [ ] Support more base models such as pi0.5
...

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/OpenHelix-Team/VLA-RFT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OpenHelix-Team/VLA-RFT/discussions)
- **Email**: [your-email@domain.com]

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:
- [VLA-Adapter](https://github.com/vla-adapter/vla-adapter) for the foundation VLA adapter model
- [VERL](https://github.com/volcengine/verl) for the RL training framework  
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) for evaluation benchmarks

---

**⭐ Star this repository if you find it helpful!**