# B.Tech AI Tutor 7B 🎓🤖

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-orange)](https://huggingface.co/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/)

**B.Tech AI Tutor 7B** is a fine-tuned, specialized LLM built on top of **Qwen 2.5-7B-Instruct**. It is designed to act as the ultimate personal study buddy and academic tutor for students pursuing a 4-year B.Tech in Computer Science & Engineering with a specialization in **Artificial Intelligence & Machine Learning (AI & ML)**. 

The model is highly optimized to run locally on resource-constrained hardware (e.g. laptops with **4GB VRAM** and **16GB System RAM**) using GGUF quantization.

---

## 🌟 Key Capabilities

- 📚 **Comprehensive Curriculum Coverage**: Expert in core subjects spanning all 8 semesters (Operating Systems, DBMS, Networks, TOC, Compiler, Architecture, etc.).
- 🧠 **AI/ML Core & Specialization**: In-depth understanding of Machine Learning theory, Deep Learning architectures, Natural Language Processing, Computer Vision, and Reinforcement Learning.
- 📐 **Rigorous Mathematical Reasoning**: Solves calculus, linear algebra, discrete math, probability & statistics problems with detailed, LaTeX-formatted derivations.
- 💻 **DSA & Coding Mastery**: Writes clean, commented, and optimal code in Python, C, C++, and Java. Debugs and explains algorithm complexities step-by-step.
- 📝 **Exam & Interview Prep**: Specialized in solving GATE CS questions, LeetCode challenges, System Design scenarios, and answering University Viva/Oral boards.

---

## 🛠️ Data Architecture (5-Layer Stack)

The model is trained using a highly structured **5-Layer Data Architecture** containing ~480K high-signal instruction-tuning samples and 15K preference pairs:

| Layer | Type | Target Samples | Source |
|---|---|---|---|
| **Layer 5** | Alignment (DPO) | ~15K pairs | Custom generated academic and code preference pairs |
| **Layer 4** | Exam & Interview Prep | ~60K samples | Custom generated GATE, ML interview, DSA, System Design, and Viva Q&As |
| **Layer 3** | CS & AI/ML Core Theory | ~150K samples | SciQ, ScienceQA, QASPER, SciTLDR, MMLU CS, Custom CS, ML, and Advanced AI Q&As |
| **Layer 2** | Mathematics & Coding | ~200K samples | OpenMathInstruct-2, MetaMathQA, MathInstruct, Magicoder, CodeAlpaca, Glaive Code, etc. |
| **Layer 1** | General Instruction Base | ~70K samples | Open-Platypus, No Robots, SlimOrca, Capybara |

---

## 🚀 Quick Start (Local Run via Ollama)

The easiest way to run the tutor model locally on your laptop is using [Ollama](https://ollama.com):

1. **Download Ollama** for your operating system.
2. **Download the GGUF file** (e.g. Q4_K_M version) from your HuggingFace Repository.
3. Place the GGUF file in your project directory and create a `Modelfile` (see `configs/Modelfile`).
4. **Build and run** the model:
   ```bash
   ollama create B.Tech-AI-Tutor-7B -f configs/Modelfile
   ollama run B.Tech-AI-Tutor-7B
   ```

---

## 🏋️ Training Pipeline

Fine-tuning is split into 3 distinct sequential training phases designed to fit within free GPU allocations (like Google Colab's T4):

### Phase 1: Core SFT (Layers 1+2+3)
- **Data**: ~420K samples
- **Hyperparameters**: Rank 128, Alpha 256, LR 2e-4, 2 epochs, batch 64.
- **Output**: Core knowledge base.

### Phase 2: Exam & Interview Specialization (Layer 4)
- **Data**: ~60K samples
- **Hyperparameters**: Rank 64, Alpha 128, LR 5e-5 (lowered to prevent forgetting), 3 epochs, batch 32.
- **Output**: Specialized exam solver.

### Phase 3: Preference Alignment (Layer 5 DPO)
- **Data**: ~15K DPO pairs
- **Hyperparameters**: Rank 32, Alpha 64, LR 5e-6, 1 epoch, Beta 0.1.
- **Output**: Production-aligned pedagogical tutor.

---

## 💻 Hardware Guidelines

### Training
- Recommended: **Google Colab (Free T4 16GB GPU)**.
- Setup script runs completely within FP16 precision (BF16 disabled since T4 does not support it).
- Google Drive is used for persistent checkpoint storage between sessions.

### Local Inference (Laptop)
- **GPU Mode (VRAM ≤ 4GB)**: Use the `Q3_K_M` GGUF quantization (~3.5 GB file size). Runs at 8-12 tokens/sec.
- **Hybrid Mode (4GB VRAM + RAM)**: Use the `Q4_K_M` GGUF quantization (~4.7 GB file size) with ~15 layers offloaded to GPU. Runs at 5-8 tokens/sec.
- **CPU Mode (16GB RAM)**: Use the `Q5_K_M` GGUF quantization (~5.5 GB file size) running entirely on CPU. Runs at 2-4 tokens/sec.

---

## 📂 Project Structure

```
finetune/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── notebooks/                          # Jupyter Notebooks for Google Colab
│   ├── 01_data_preparation.ipynb       # Downloads & formats datasets
│   ├── 02_phase1_sft.ipynb             # Phase 1 SFT core training
│   ├── 03_phase2_sft.ipynb             # Phase 2 SFT exam/interview tuning
│   ├── 04_phase3_dpo.ipynb             # Phase 3 DPO alignment training
│   └── 05_evaluate_quantize.ipynb      # Evaluates weights & converts to GGUF
├── scripts/                            # Core pipeline Python scripts
│   ├── download_datasets.py            # Pulls public HF datasets
│   ├── generate_custom_data.py         # Generates custom curriculum Q&A
│   ├── create_dpo_pairs.py             # Generates preference triplet data
│   ├── quality_filter.py               # LSH MinHash cleaning & deduplication
│   └── format_and_merge.py             # Prepares final ChatML train sets
├── configs/                            # Deployment configurations
│   ├── Modelfile                       # Ollama Modelfile configuration
│   └── model_card.md                   # Model metadata for HuggingFace
└── docs/
    └── setup_guide.md                  # Laptop GGUF deployment instructions
```

---

## 📄 License
This project is licensed under the Apache License 2.0. Base model weights are licensed under the Qwen License.

## 🤝 Acknowledgments
- Thanks to the Alibaba **Qwen Team** for the base model.
- Thanks to **Unsloth** for making fine-tuning fast and resource-efficient.
