---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- curriculum
- educational
- logic
- machine learning
- computer science
- code
- math
- unsloth
- dpo
- qlora
---

# B.Tech AI Tutor 7B

**B.Tech AI Tutor 7B** is a fine-tuned version of Alibaba's **Qwen 2.5-7B-Instruct** model, specifically optimized to serve as an expert academic companion for students enrolled in a 4-year B.Tech program in Computer Science & Engineering specializing in Artificial Intelligence and Machine Learning (AI & ML).

## Model Description

This model was trained across a highly structured 5-layer academic and coding dataset using QLoRA and preference alignment (DPO) on T4 GPU architectures. It has deep domain coverage in:
1. **Core Computer Science**: Operating Systems, Database Management Systems (DBMS), Computer Networks, Theory of Computation (TOC), Compiler Design, and Computer Organization and Architecture (COA).
2. **AI & ML Specialization**: Machine Learning, Deep Learning, Natural Language Processing (NLP), Computer Vision (CV), Reinforcement Learning (RL), and MLOps.
3. **Engineering Mathematics**: Calculus, Linear Algebra, Probability & Statistics, Optimization, and Discrete Mathematics.
4. **Programming & Algorithms**: Data Structures & Algorithms (DSA), competitive programming, and systems coding in C, C++, Python, and Java.
5. **Exam & Interview Preparation**: Detailed walkthroughs for GATE CS problems, LeetCode assessments, System Design, and university viva preparations.

## Training Procedure

The model was trained in 3 distinct phases using **Unsloth**:
- **Phase 1 (SFT)**: Shuffled training on Layers 1, 2, and 3 (~420K samples) focusing on general instructions, math, code, and academic CS/ML concepts. (LoRA rank=128, alpha=256, learning_rate=2e-4, 2 epochs).
- **Phase 2 (SFT)**: Tuning on Layer 4 (~60K samples) focusing on exams, placement coding, and viva interviews. (LoRA rank=64, alpha=128, learning_rate=5e-5, 3 epochs).
- **Phase 3 (DPO)**: Direct Preference Optimization on Layer 5 (~15K preference pairs) prioritizing detailed, structured, and pedagogical responses. (LoRA rank=32, alpha=64, learning_rate=5e-6, 1 epoch).

## Intended Use

The model is packaged in GGUF formats (`Q3_K_M`, `Q4_K_M`, `Q5_K_M`) for local deployment on standard consumer laptops with 4GB VRAM and 16GB System RAM using Ollama or LM Studio. It functions best as an interactive study buddy, textbook explainer, code debugger, or test prep coach.

### Sample System Prompt
```
You are BTech-AI-Tutor, a highly specialized AI tutor designed to help students studying B.Tech Computer Science & Engineering with specialization in AI & ML. Explain complex CS concepts clearly, provide step-by-step math derivations in LaTeX, write commented code, and explain runtime complexities.
```

## Limitations & Bias

While B.Tech AI Tutor 7B is highly optimized for academic CS, coding, and mathematical reasoning, it is still subject to LLM hallucinations on extremely niche or unrepresented topics. Always verify critical compiler output, mathematical proofs, or system design topologies before applying them to production contexts.
