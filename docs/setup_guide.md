# Local Inference Setup Guide for Laptop (4GB VRAM + 16GB RAM)

This guide walks you through setting up and running your fine-tuned **B.Tech AI Tutor 7B** model locally on a consumer laptop. The guide covers different configurations optimized to run smoothly on hardware with **4GB of Dedicated VRAM** and **16GB of System RAM**.

---

## Deployment Option 1: Ollama (Recommended)

Ollama is the easiest tool for local execution as it automatically manages CPU/GPU layers, supports context-window customization, and runs a local background API.

### Step 1: Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com).

### Step 2: Download GGUF Files
Download the GGUF model files (e.g. `btech-ai-tutor-q4_k_m.gguf`) from your HuggingFace repository or Google Drive, and place it in a known folder.

### Step 3: Create a Modelfile
In the same folder, verify your `Modelfile` (available at `configs/Modelfile`). If you downloaded the `Q3_K_M` or `Q4_K_M` file, update the `FROM` path in the Modelfile:
```dockerfile
FROM ./btech-ai-tutor-q4_k_m.gguf
```

### Step 4: Build and Run the Model
Open your terminal/command prompt and run:
```bash
# Build the model in Ollama
ollama create btech-ai-tutor -f Modelfile

# Start the interactive chat session
ollama run btech-ai-tutor
```

---

## Deployment Option 2: LM Studio (User-Friendly UI)

If you prefer a clean graphical user interface (GUI) with side-by-side chats, parameter sliders, and system prompt configurations:

1. Download and install **LM Studio** from [lmstudio.ai](https://lmstudio.ai).
2. Open LM Studio, click on the **Search Icon** on the left panel, and search for your HuggingFace model repo (e.g. `username/btech-ai-tutor-7b-GGUF`).
3. Download either the `Q3_K_M` (fits entirely on 4GB VRAM) or `Q4_K_M` (recommended balance) model.
4. Go to the **Chat** panel, select your model from the top dropdown.
5. In the **Right Sidebar**, configure:
   - **GPU Acceleration**: Set "GPU Offload" (Hardware Settings) to around **15-20 layers**. This ensures part of the model runs on your GPU (within 4GB VRAM limit) and the rest runs on CPU, maximizing generation speed.
   - **Context Length**: Set context length to **4096**.
   - **System Prompt**: Paste the system prompt from `configs/Modelfile`.

---

## Deployment Option 3: llama.cpp (For Performance Geeks)

For direct console execution without background services or heavy wrappers:

### For CPU-only Mode (Runs entirely on 16GB System RAM - High Accuracy)
Using the `Q5_K_M` model on CPU guarantees maximum accuracy but yields slower inference speeds (2-4 tokens/sec).
```bash
# Run with 8 CPU threads and 4096 context length
./llama-cli \
    -m path/to/btech-ai-tutor-q5_k_m.gguf \
    -ngl 0 \
    -c 4096 \
    -t 8 \
    --interactive-first \
    -p "You are BTech-AI-Tutor..."
```

### For Hybrid GPU+CPU Mode (Fits on 4GB VRAM + RAM - Balanced)
```bash
# Offload 15 layers to GPU, utilize 6 CPU threads
./llama-cli \
    -m path/to/btech-ai-tutor-q4_k_m.gguf \
    -ngl 15 \
    -c 4096 \
    -t 6 \
    --interactive-first \
    -p "You are BTech-AI-Tutor..."
```

---

## 💡 Troubleshooting & Performance Tips

### 1. Slow response times (low tokens/sec)
- **VRAM Overflow**: If your GPU offload is too high, you might overflow your 4GB VRAM, forcing the system to use shared memory (which is extremely slow). Reduce the GPU offloaded layers by 2-3 layers in LM Studio or llama.cpp.
- **Background Processes**: Close heavy browser tabs, game launchers, or development environments to free up system RAM.
- **Thread Count**: If running CPU mode, match the threads (`-t`) to your CPU's physical core count (usually 6 or 8), not virtual threads.

### 2. "Out of memory" errors
- Make sure you are using either the **Q3_K_M** or **Q4_K_M** model. Do not try to run Q8_0 or FP16 models on a 4GB VRAM laptop, as they will immediately crash the application.
- If using Ollama, close LM Studio (and vice-versa) to prevent resource conflicts.
