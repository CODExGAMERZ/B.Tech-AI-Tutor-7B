# HuggingFace Setup & Model Upload Guide

This guide walks you through setting up your HuggingFace account, generating API tokens, preparing repositories, and pushing your datasets and model checkpoints during training.

---

## Step 1: Create a HuggingFace Account
1. Go to [huggingface.co](https://huggingface.co/) and click **Sign Up**.
2. Complete your profile registration.

---

## Step 2: Generate an API Access Token
HuggingFace uses access tokens to authenticate writes/uploads from Colab or script terminals.

1. Navigate to your **Settings** (click your profile picture in the top-right → **Settings**).
2. On the left sidebar, click **Access Tokens**.
3. Click **New Token**.
4. Set the name (e.g. `colab-tutor-training`).
5. **CRITICAL**: Set the role to **Write** (otherwise you won't be able to upload models or datasets).
6. Click **Generate a token** and copy it somewhere safe.

---

## Step 3: Add the Token to Google Colab
To avoid hardcoding secrets in your notebooks:

1. Open your Google Colab notebook.
2. Click the **Key Icon** (Secrets) on the left sidebar.
3. Add a new secret:
   - **Name**: `HF_TOKEN`
   - **Value**: paste your HuggingFace write token.
4. Toggle on **Notebook access** for `HF_TOKEN`.

---

## Step 4: Create Repositories on HuggingFace
You will need repositories on HuggingFace to store your custom datasets and your final GGUF models.

### Option A: Create via Website (Easiest)
1. To create a model repo: Go to [huggingface.co/new](https://huggingface.co/new).
   - Set **Owner** to your username.
   - Set **Model name** (e.g. `btech-ai-tutor-7b-GGUF` or `btech-ai-tutor-7b-adapters`).
   - Select **Public** or **Private**.
   - Click **Create model**.
2. To create a dataset repo: Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset).
   - Set **Dataset name** (e.g. `btech-ai-tutor-custom-data`).
   - Click **Create dataset**.

### Option B: Create Programmatically (in Python)
The training notebooks will try to create these automatically if you provide your username. Here is the code they run:
```python
from huggingface_hub import HfApi
api = HfApi()

# Create model repo
api.create_repo(
    repo_id="YOUR_USERNAME/btech-ai-tutor-7b-GGUF",
    repo_type="model",
    exist_ok=True
)
```

---

## Step 5: How the Uploads Work

### 1. Uploading Datasets (Phase 1 Setup)
In notebook `01_data_preparation.ipynb`, after generating custom Q&A datasets, you can push them directly to a HuggingFace Dataset repository so that you can easily load them during SFT training:
```python
from datasets import load_dataset
# Load from local JSONL
dataset = load_dataset("json", data_files="data/processed/phase1_train.jsonl")
# Push directly to HuggingFace Datasets hub
dataset.push_to_hub("YOUR_USERNAME/btech-ai-tutor-dataset")
```

### 2. Uploading adapters during training (Automatic checkpointers)
In notebooks `02_phase1_sft.ipynb` and `03_phase2_sft.ipynb`, we set up `push_to_hub=True` in the HuggingFace `TrainingArguments`. This automatically uploads your training progress and final adapter weights to the hub:
```python
training_args = TrainingArguments(
    output_dir="./checkpoints",
    push_to_hub=True,
    hub_model_id="YOUR_USERNAME/btech-ai-tutor-7b-adapter",
    hub_strategy="checkpoint",
    # ... other args
)
```

### 3. Uploading Final GGUF Models
In notebook `05_evaluate_quantize.ipynb`, we quantize our final merged model to GGUF format and upload the GGUF folders to our GGUF repo:
```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/content/drive/MyDrive/btech-ai-tutor/models/gguf_q4",
    repo_id="YOUR_USERNAME/btech-ai-tutor-7b-GGUF",
    path_in_repo="q4_k_m"
)
```
Once uploaded, you can click on the `Files and versions` tab of your HuggingFace repository web page to download the GGUF files to your local laptop!
