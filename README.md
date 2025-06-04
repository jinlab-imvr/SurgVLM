# SurgVLM: A Large Vision-Language Model and Systematic Evaluation Benchmark for Surgical Intelligence

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.02555-b31b1b.svg)](https://www.arxiv.org/abs/2506.02555)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://jinlab-imvr.github.io/SurgVLM/)
[![GitHub](https://img.shields.io/github/stars/jinlab-imvr/SurgVLM?style=social)](https://github.com/jinlab-imvr/SurgVLM)

**[🌐 Project Page](https://jinlab-imvr.github.io/SurgVLM/) | [📄 Paper](https://www.arxiv.org/abs/2506.02555) | [🤗 Models](https://github.com/jinlab-imvr/SurgVLM) | [📊 Leaderboard](https://jinlab-imvr.github.io/SurgVLM/)**

*Building the future of surgical intelligence with AI* 🏥🤖

</div>

## 📖 Overview

Foundation models have achieved transformative success across biomedical domains by enabling holistic understanding of multimodal data. However, their application in surgery remains underexplored. Surgical intelligence presents unique challenges - requiring surgical visual perception, temporal analysis, and reasoning. 

**SurgVLM** is one of the first large vision-language foundation models for surgical intelligence, where this single universal model can tackle versatile surgical tasks. We propose:

- 🗃️ **SurgVLM-DB**: A large-scale multimodal surgical database comprising over **1.81 million frames** with **7.79 million conversations**
- 🏆 **SurgVLM-Bench**: A comprehensive surgical benchmark for vision-language models evaluation  
- 🚀 **SurgVLM Models**: Three model variants (SurgVLM-7B, SurgVLM-32B, SurgVLM-72B) achieving state-of-the-art performance

## 🚀 Key Features

### 🎯 SurgVLM-DB: Large-scale Surgical Multimodal Database
- **1.81M+ annotated surgical images** with **7.79M+ conversations**
- **16 surgical types** and **18 anatomical structures**
- **23 public datasets** unified across **10 surgical tasks**
- Hierarchical vision-language alignment from visual perception to high-level reasoning

### 📊 SurgVLM-Bench: Comprehensive Surgical Benchmark
- **6 popular surgical datasets** covering crucial downstream tasks
- Evaluation across multiple surgical intelligence domains:
  - Phase Recognition
  - Action Recognition  
  - Triplet Prediction
  - Instrument Localization
  - Critical View Safety Detection
  - Multi-task VQA

### 🤖 SurgVLM Models: State-of-the-Art Performance
- **SurgVLM-7B**: Built on Qwen2.5-7B with DFN CLIP vision encoder
- **SurgVLM-32B**: Built on Qwen2.5-32B with DFN CLIP vision encoder  
- **SurgVLM-72B**: Built on Qwen2.5-72B with DFN CLIP vision encoder

## 📈 Performance

Our SurgVLM-72B achieves **75.4% improvement** on overall arena score compared with Gemini 2.0 Flash:
- **96.5%** improvement in phase recognition
- **87.7%** improvement in action recognition  
- **608.1%** improvement in triplet prediction
- **198.5%** improvement in instrument localization
- **28.9%** improvement in critical view safety detection
- **59.4%** improvement in comprehensive multi-task VQA

## 🏆 Leaderboard

| Rank | Model | Arena Score ↑ | Phase Acc | Action Acc | Triplet Acc | CVS Acc | VQA Acc | Loc mIoU |
|------|-------|---------------|-----------|------------|-------------|---------|---------|----------|
| 🥇 1  | SurgVLM-72B (Ours) | **336.21** | **69.66** | **43.1** | **12.52** | **76.73** | **75.2** | **59.0** |
| 🥈 2  | SurgVLM-72B (Ours) | **331.86** | **76.40** | **42.9** | **13.10** | **76.60** | **63.46** | **59.4** |
| 🥉 3  | SurgVLM-32B (Ours) | **306.91** | **71.20** | **40.1** | **12.98** | **74.51** | **59.72** | **48.4** |
| 4  | SurgVLM-7B (Ours) | **290.78** | **70.30** | **45.8** | **4.15** | **76.86** | **59.67** | **34.0** |
| 5  | Gemini 2.0 Flash | 191.70 | 38.89 | 24.4 | 1.85 | 59.61 | 47.05 | 19.9 |

*Complete leaderboard with 20 models available on our [project page](https://jinlab-imvr.github.io/SurgVLM/)*

## 📦 Model Downloads

| Model | Size | HuggingFace | Description |
|-------|------|-------------|-------------|
| SurgVLM-7B | 7B | 🤗 [Coming Soon] | Full-tuning variant with DFN CLIP |
| SurgVLM-32B | 32B | 🤗 [Coming Soon] | Freeze-tuning variant with DFN CLIP |
| SurgVLM-72B | 72B | 🤗 [Coming Soon] | LoRA-tuning variant with DFN CLIP |

## 📊 Dataset Downloads

| Dataset | Size | Download | Description |
|---------|------|----------|-------------|
| SurgVLM-DB | 1.81M images, 7.79M conversations | 🤗 [Coming Soon] | Complete surgical multimodal database |
| SurgVLM-Bench | 6 datasets | 🤗 [Coming Soon] | Evaluation benchmark for surgical VLMs |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/jinlab-imvr/SurgVLM.git
cd SurgVLM

# Install dependencies
pip install -r requirements.txt

# Download models (links will be updated soon)
# python download_models.py
```

## 🚀 Quick Start

```python
# Example usage (code will be released soon)
from surgvlm import SurgVLM

# Load model
model = SurgVLM.from_pretrained("SurgVLM-7B")

# Surgical image analysis
image_path = "path/to/surgical/image.jpg"
question = "What surgical instruments are visible in this image?"

response = model.generate(image_path, question)
print(response)
```

## 📋 Supported Tasks

SurgVLM supports **10 surgical tasks** across three categories:

### 🎯 Visual Perception
- **Instrument Recognition**: Identify surgical instruments
- **Instrument Localization**: Locate instruments with bounding boxes
- **Instrument Location**: Determine spatial regions of instruments
- **Tissue Recognition**: Recognize anatomical tissues
- **Tissue Localization**: Locate tissues with precise coordinates

### ⏱️ Temporal Analysis  
- **Phase Recognition**: Identify surgical phases
- **Step Recognition**: Recognize procedural steps
- **Action Recognition**: Classify surgical actions

### 🧠 High-level Reasoning
- **Triplet Recognition**: Understand instrument-action-tissue relationships
- **Critical View of Safety**: Evaluate safety criteria in laparoscopic surgery

## 📊 Evaluation

Evaluate your model on SurgVLM-Bench:

```bash
# Evaluation script (coming soon)
python evaluate.py --model_path /path/to/model --benchmark surgvlm_bench
```

## 🏗️ Training

Train your own SurgVLM model:

```bash
# Training script (coming soon)
python train.py --config configs/surgvlm_7b.yaml
```

## 👥 Team

This work is a collaboration between:

- **National University of Singapore** 🇸🇬
- **Nanyang Technological University** 🇸🇬  
- **University of Oxford** 🇬🇧
- **State Key Laboratory of General Artificial Intelligence, BIGAI** 🇨🇳
- **Shanghai Jiao Tong University** 🇨🇳
- **Sun Yat-sen University** 🇨🇳
- **The Chinese University of Hong Kong** 🇭🇰

## 📄 Citation

If you find SurgVLM useful for your research, please cite our paper:

```bibtex
@misc{zeng2025surgvlm,
    title={SurgVLM: A Large Vision-Language Model and Systematic Evaluation Benchmark for Surgical Intelligence},
    author={Zhitao Zeng and Zhu Zhuo and Xiaojun Jia and Erli Zhang and Junde Wu and Jiaan Zhang and Yuxuan Wang and Chang Han Low and Jian Jiang and Zilong Zheng and Xiaochun Cao and Yutong Ban and Qi Dou and Yang Liu and Yueming Jin},
    year={2025},
    eprint={2506.02555},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## 📧 Contact

For questions about SurgVLM, please contact:
- **Zhitao Zeng**: zhitao@nus.edu.sg
- **Yueming Jin**: yueming.jin@nus.edu.sg

## 🙏 Acknowledgments

We gratefully acknowledge the support from all institutions and contributors. Special thanks to the surgical community for providing valuable datasets and feedback.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
