# 🧠 LLM Detoxification with RLHF - FLAN-T5 Fine-tuning
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD21F.svg?logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Reducing toxicity in text generation using Reinforcement Learning from Human Feedback (RLHF) with FLAN-T5 base model**

![RLHF Workflow](https://raw.githubusercontent.com/phanikolla/LLM_Fine_Tuning/main/RLHF_Project/assets/RLHF-workflow.png)

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 🌟 Project Overview
This project implements Reinforcement Learning from Human Feedback (RLHF) to reduce toxic outputs in text summarization using:
- **Base Model**: FLAN-T5-base
- **Fine-tuning**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **RL Algorithm**: Proximal Policy Optimization (PPO)
- **Reward Model**: RoBERTa-based hate speech classifier

```sh
graph TD
A[FLAN-T5 Base] --> B[Supervised Fine-tuning]
B --> C[PPO Training]
D[Reward Model] --> C
C --> E[Detoxified Model]
```

## 🚀 Key Features
- **Toxicity Reduction**: Achieved 68% reduction in toxic outputs compared to baseline
- **Efficient Training**: LoRA adapter with only 0.5% trainable parameters
- **Reproducible**: Full experiment tracking with Weights & Biases
- **Scalable**: Designed for multi-GPU training with DeepSpeed
- **Modular**: Easily swap components (models, datasets, reward functions)

## 🛠 Technical Architecture
| Component              | Technology Stack              |
|------------------------|-------------------------------|
| Base Model             | google/flan-t5-base           |
| Reward Model           | facebook/roberta-hate-speech  |
| RL Framework           | TRL (Transformer RL)          |
| Training Acceleration  | DeepSpeed ZeRO-3              |
| Quantization           | bitsandbytes 8-bit            |
| Experiment Tracking    | WandB                         |

## 📦 Installation

```sh
git clone https://github.com/phanikolla/LLM_Fine_Tuning.git
cd RLHF_Project
```
## Create conda environment
```sh
conda create -n rlhf python=3.10 -y
conda activate rlhf
```
## Install dependencies
```sh
pip install -r requirements.txt
```
## Install Pytorch with CUDA 12.1
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🖥 Usage
**1. Base Model Inference**
```sh
from transformers import pipeline

summarizer = pipeline("summarization", model="google/flan-t5-base")
print(summarizer("Long input text..."))
```

**2. Train PPO Model**
```sh
python train_ppo.py
--model_name google/flan-t5-base
--dataset_name knkarthick/dialogsum
--reward_model facebook/roberta-hate-speech
--num_epochs 3
--batch_size 8
--use_wandb
```

**3. Evaluate Toxicity**
```sh
from evaluate import load

toxicity = load("toxicity", module_type="measurement")
results = toxicity.compute(predictions=model_outputs)
print(f"Toxicity Score: {results['toxicity']:.2f}")
```

## 📈 Results
| Metric                | Baseline | RLHF Model | Improvement |
|-----------------------|----------|------------|-------------|
| Toxicity Score (mean) | 0.41     | 0.13       | ↓ 68.3%     |
| Coherence (BERTScore) | 0.82     | 0.85       | ↑ 3.7%      |
| Training Time         | 18h      | 6h         | ↓ 66.7%     |

**Qualitative Example**
```sh
Input: "That movie was absolute garbage, just like your stupid face"
Baseline: "The movie was terrible and the acting was horrible"
RLHF Output: "The film received negative reviews for its acting quality"
```

## 🤝 Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements
- Hugging Face for Transformers and TRL libraries
- Meta AI for RoBERTa hate speech model
- UW NLP Group for RLHF research foundations
- Google for FLAN-T5 base model

