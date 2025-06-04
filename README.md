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

| Rank | Model | Institute | Evaluation | Arena Score ↑ | Phase Acc | Action Acc | Triplet Acc | CVS Acc | VQA Acc | Loc mIoU |
|------|-------|-----------|------------|---------------|-----------|------------|-------------|---------|---------|----------|
| 🥇 1  | SurgVLM-72B (Ours) | iMVR Lab | MCQ | **336.21** | **69.66** | **43.1** | **12.52** | **76.73** | **75.2** | **59.0** |
| 🥈 2  | SurgVLM-72B (Ours) | iMVR Lab | OV | **331.86** | **76.40** | **42.9** | **13.10** | **76.60** | **63.46** | **59.4** |
| 🥉 3  | SurgVLM-32B (Ours) | iMVR Lab | OV | **306.91** | **71.20** | **40.1** | **12.98** | **74.51** | **59.72** | **48.4** |
| 4  | SurgVLM-7B (Ours) | iMVR Lab | OV | **290.78** | **70.30** | **45.8** | **4.15** | **76.86** | **59.67** | **34.0** |
| 5  | Gemini 2.0 Flash | Google DeepMind | MCQ | 191.70 | 38.89 | 24.4 | 1.85 | 59.61 | 47.05 | 19.9 |
| 6  | Qwen2.5-VL-72B-Instruct | Alibaba Cloud | MCQ | 184.85 | 29.30 | 28.2 | 1.27 | 41.69 | 42.19 | 42.2 |
| 7  | Qwen2.5-VL-32B-Instruct | Alibaba Cloud | MCQ | 184.40 | 37.23 | 31.8 | 0.98 | 60.53 | 42.46 | 11.4 |
| 8  | Qwen2.5-VL-7B-Instruct | Alibaba Cloud | MCQ | 175.20 | 30.45 | 31.1 | 0.35 | 65.88 | 36.82 | 10.6 |
| 9  | Qwen 2.5 Max | Alibaba Cloud | MCQ | 174.37 | 34.79 | 28.3 | 0.35 | 34.77 | 36.16 | 40.0 |
| 10 | InternVL3-78B | Shanghai AI Lab | MCQ | 172.97 | 27.32 | 29.5 | 0.52 | 50.20 | 36.33 | 29.1 |
| 11 | Llama-4-Scout-17B-16E-Instruct | Meta AI | MCQ | 163.84 | 35.77 | 25.1 | 0.58 | 37.39 | 37.00 | 28.0 |
| 12 | Mistral-Small-3.1-24B-Instruct-2503 | Mistral AI | MCQ | 156.98 | 22.61 | 12.5 | 0.46 | 68.10 | 36.41 | 16.9 |
| 13 | InternVL3-8B | Shanghai AI Lab | MCQ | 146.42 | 23.88 | 29.3 | 2.08 | 48.24 | 34.72 | 8.2 |
| 14 | MiniCPM-O-2_6 | ModelBest | MCQ | 140.34 | 17.75 | 30.8 | 0.06 | 35.95 | 35.48 | 20.3 |
| 15 | Gemma3-27B-it | Google DeepMind | MCQ | 138.93 | 14.08 | 33.2 | 0.06 | 38.04 | 35.95 | 17.6 |
| 16 | Phi-4-Multimodal-Instruct | Microsoft | MCQ | 131.10 | 22.45 | 15.1 | 0.12 | 58.43 | 34.20 | 0.8 |
| 17 | MiniCPM-V-2_6 | MiniCPM Team | MCQ | 128.77 | 15.20 | 24.3 | 0 | 38.69 | 33.28 | 17.3 |
| 18 | GPT-4o | OpenAI | MCQ | 118.71 | 36.43 | 28.1 | 1.50 | 6.67 | 38.31 | 7.7 |
| 19 | LLava-1.5-7B | WAIV Lab | MCQ | 112.57 | 23.46 | 5.1 | 0 | 25.49 | 31.42 | 27.1 |
| 20 | Skywork-R1V-38B | Skywork AI | MCQ | 107.64 | 6.37 | 12.3 | 0 | 43.79 | 34.58 | 10.6 |

**Note**: The leaderboard is ranked by the **Arena Score** obtained by summing the most important metrics across six surgical tasks. Higher scores indicate better performance. MCQ = Multiple Choice Questions, OV = Open Vocabulary.

*Submit your results to: zhitao@nus.edu.sg*

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
