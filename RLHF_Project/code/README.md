# 🧠 LLM Detoxification with RLHF - FLAN-T5 Fine-tuning
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD21F.svg?logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Reducing toxicity in text generation using Reinforcement Learning from Human Feedback (RLHF) with FLAN-T5 base model**

![RLHF Workflow](https://raw.githubusercontent.com/phanikolla/LLM_Fine_Tuning/main/RLHF_Project/assets/workflow.png)

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
