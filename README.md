# SurgVLM: A Unified Vision-Language Framework for Surgical Intelligence

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.02555-b31b1b.svg)](https://www.arxiv.org/abs/2506.02555)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://jinlab-imvr.github.io/SurgVLM/)
[![GitHub](https://img.shields.io/github/stars/jinlab-imvr/SurgVLM?style=social)](https://github.com/jinlab-imvr/SurgVLM)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-SurgSigma--DB-yellow.svg)](https://huggingface.co/datasets/SurgSigma/SurgSigma-DB)

### **[🌐 Project Page](https://jinlab-imvr.github.io/SurgVLM/) · [📄 Paper](https://www.arxiv.org/abs/2506.02555) · [🤗 Dataset](https://huggingface.co/datasets/SurgSigma/SurgSigma-DB) · [🤗 Models](https://github.com/jinlab-imvr/SurgVLM)**

*Advancing surgical intelligence with vision-language models* 🏥🤖

</div>

---

## 📖 Overview

Surgical intelligence requires more than general visual understanding. It involves perception of surgical scenes, temporal understanding of procedural progress, and reasoning over instruments, anatomy, actions, and safety.

**SurgVLM** is a unified vision-language framework designed for surgical intelligence. It supports diverse surgical tasks within a single modeling pipeline, spanning visual perception, temporal analysis, and high-level reasoning.

Our project includes:

- 🗃️ **SurgVLM-DB**: a surgical multimodal corpus for training surgical vision-language models
- 🤖 **SurgVLM Models**: two specialized variants for instruction following and reasoning
- 🚀 **Unified Surgical Intelligence**: one framework for diverse surgical understanding tasks

---

## ✨ Highlights

- Built specifically for **surgical intelligence**
- Supports **visual perception**, **temporal understanding**, and **reasoning**
- Covers **10 surgical tasks**
- Includes two specialized models built on **Qwen3.5-9B**
- Designed for **general surgical understanding** and **complex reasoning scenarios**
- **Part of the training data is publicly available on Hugging Face**

---

## 🗃️ SurgVLM-DB

**SurgVLM-DB** is a multimodal surgical corpus designed for training domain-specific vision-language models.

### Key Characteristics

- Integrates diverse surgical data sources
- Covers multiple surgical procedures, anatomical structures, and task types
- Supports model training from low-level perception to high-level reasoning
- Provides a unified foundation for surgical vision-language learning

### Public Release

Part of the training data has been publicly released through **SurgSigma-DB** on Hugging Face:

- **Hugging Face**: [SurgSigma/SurgSigma-DB](https://huggingface.co/datasets/SurgSigma/SurgSigma-DB)

Additional resources and updates will be released through the project page and repository.

---

## 🤖 Models

SurgVLM provides two specialized model variants, both built on **Qwen3.5-9B**, to support different surgical intelligence scenarios.

| Model | Backbone | Description |
|-------|----------|-------------|
| **SurgVLM-9B-Instruct** | Qwen3.5-9B | An instruction-tuned model for general surgical vision-language understanding |
| **SurgVLM-9B-Reasoning** | Qwen3.5-9B | A reasoning-oriented model for more complex surgical analysis and decision-related tasks |

---

## 📈 Capabilities

SurgVLM is designed to support a broad range of surgical intelligence tasks, including:

- **Phase recognition**
- **Step recognition**
- **Action recognition**
- **Triplet understanding**
- **Instrument localization**
- **Critical view of safety assessment**

These capabilities make SurgVLM a unified framework for surgical scene understanding and reasoning.

---

## 📦 Downloads

### Models

| Model | Backbone | Download | Status |
|-------|----------|----------|--------|
| SurgVLM-9B-Instruct | Qwen3.5-9B | 🤗 Coming Soon | To be released |
| SurgVLM-9B-Reasoning | Qwen3.5-9B | 🤗 Coming Soon | To be released |

### Data

| Resource | Download | Status | Description |
|----------|----------|--------|-------------|
| SurgVLM-DB | [🤗 Hugging Face](https://huggingface.co/datasets/SurgSigma/SurgSigma-DB) | Partially released | Surgical multimodal corpus for model training |

---

## 🛠️ Installation

```bash
git clone https://github.com/jinlab-imvr/SurgVLM.git
cd SurgVLM
pip install -r requirements.txt
