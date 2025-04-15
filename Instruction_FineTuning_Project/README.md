# 🚀 LLM Fine-Tuning Toolkit
### Instruction Fine-Tuning for Modern Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![LLM Fine-Tuning Banner](https://raw.githubusercontent.com/phanikolla/LLM_Fine_Tuning/main/Instruction_FineTuning_Project/llm_fineTuning.png)

> **Professional-grade toolkit for instruction fine-tuning of large language models**

## 📑 Table of Contents
- [✨ Overview](#-overview)
- [🚀 Features](#-features)
- [⚙️ Installation](#️-installation)
- [💻 Usage](#-usage)
- [📊 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## ✨ Overview
This repository contains a comprehensive toolkit for instruction fine-tuning of large language models (LLMs) using modern parameter-efficient techniques. Built with researchers and practitioners in mind, it supports:

- 🔄 **Instruction Fine-Tuning** for task-specific adaptation
- 🧠 **LoRA (Low-Rank Adaptation)** for efficient parameter updates
- ⚡ **QLoRA Quantization** for memory-efficient training
- 📈 **Full Model Fine-Tuning** capabilities

**Key Architecture**:
```sh
graph TD
A[Base Model] --> B[Instruction Dataset]
B --> C[LoRA Adapters]
C --> D[Fine-Tuned Model]
D --> E[Evaluation Metrics]
```

## 🚀 Features
- 💾 Multiple precision modes (FP32, FP16, BF16)
- 🧩 Modular design for easy customization
- 📊 Integrated evaluation metrics (BLEU, ROUGE, perplexity)
- 🐳 Docker support for reproducible environments
- 📈 TensorBoard integration for training visualization
- 🔄 Hugging Face Hub integration for model sharing

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+
- NVIDIA GPU with ≥24GB VRAM (for full fine-tuning)

### Quick Start
```sh
git clone https://github.com/phanikolla/LLM_Fine_Tuning.git
cd LLM_Fine_Tuning/Instruction_FineTuning_Project
```
### Install dependencies
```sh
pip install -r requirements.txt
```
### Prepare sample dataset
```sh
python scripts/prepare_dataset.py --config configs/dataset_config.yaml
```

## 💻 Usage

### Training
```sh
from finetuning import Trainer

trainer = Trainer(
model_name="mistralai/Mistral-7B-v0.1",
dataset="alpaca_cleaned",
use_lora=True,
lora_r=8
)

trainer.train(
epochs=3,
batch_size=4,
learning_rate=2e-5
)
```

### Inference
```sh
from inference import Generator

generator = Generator("checkpoints/final_model")
response = generator.generate("Explain quantum computing in simple terms")
print(response)
```

## 📊 Results
### Performance Comparison (Mistral-7B)

| Metric          | Base Model | Fine-Tuned |
|-----------------|------------|------------|
| Accuracy (%)    | 62.4       | 78.9       |
| Perplexity      | 24.7       | 12.3       |
| Inference Speed | 142ms      | 155ms      |

![Training Loss Curve](https://raw.githubusercontent.com/phanikolla/LLM_Fine_Tuning/main/Instruction_FineTuning_Project/trainingLoss.png)

## 🤝 Contributing
We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) and follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments
- Hugging Face team for Transformers library
- Stanford Alpaca team for dataset inspiration
- Microsoft Research for LoRA methodology

---

⭐ **Star History**  
[![Star History Chart](https://api.star-history.com/svg?repos=phanikolla/LLM_Fine_Tuning&type=Date)](https://star-history.com/#phanikolla/LLM_Fine_Tuning&Date)


