import os
import argparse
import json
import random
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an expert AI tutor for B.Tech Computer Science & Engineering with "
    "specialization in AI & ML. You provide detailed, accurate, and pedagogically "
    "sound explanations. When solving problems, show step-by-step working. When "
    "writing code, include comments and explain the logic."
)

def format_sample(instruction, output):
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": text}

def extract_fields(sample):
    """Extract instruction/output keys dynamically since HF datasets use various schemas."""
    # Find instruction key
    instr_keys = ["instruction", "question", "input", "prompt"]
    instruction = ""
    for k in instr_keys:
        if k in sample:
            instruction = sample[k]
            break
            
    # Find output key
    out_keys = ["output", "response", "answer", "completion"]
    output = ""
    for k in out_keys:
        if k in sample:
            output = sample[k]
            break
            
    # Handle specific complex datasets if needed
    if not instruction and "text" in sample:
        # Fallback if text is the only key
        instruction = sample["text"]
        
    return instruction, output

def process_file(file_path):
    samples = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return samples
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    instruction, output = extract_fields(data)
                    if instruction and output:
                        formatted = format_sample(instruction, output)
                        samples.append(formatted)
                except Exception as e:
                    print(f"Error parsing line in {file_path}: {e}")
    return samples

def main():
    parser = argparse.ArgumentParser(description="Reformat datasets into Qwen ChatML and merge for training phases")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing raw jsonl files")
    parser.add_argument("--custom_dir", type=str, default="data/custom", help="Directory containing custom generated jsonl files")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save final processed files")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # -------------------------------------------------------------
    # PHASE 1 SFT: Layers 1 (General), 2 (Math/Code), 3 (CS/AI Domain)
    # -------------------------------------------------------------
    phase1_files = [
        # Layer 1
        (args.raw_dir, "open_platypus.jsonl"),
        (args.raw_dir, "no_robots.jsonl"),
        (args.raw_dir, "slim_orca.jsonl"),
        (args.raw_dir, "capybara.jsonl"),
        # Layer 2A
        (args.raw_dir, "open_math_instruct.jsonl"),
        (args.raw_dir, "meta_math_qa.jsonl"),
        (args.raw_dir, "math_instruct.jsonl"),
        (args.raw_dir, "orca_math.jsonl"),
        (args.raw_dir, "competition_math.jsonl"),
        # Layer 2B
        (args.raw_dir, "magicoder_evol.jsonl"),
        (args.raw_dir, "code_alpaca.jsonl"),
        (args.raw_dir, "glaive_code.jsonl"),
        (args.raw_dir, "code_contests.jsonl"),
        (args.raw_dir, "self_oss_instruct.jsonl"),
        # Layer 3
        (args.raw_dir, "sciq.jsonl"),
        (args.raw_dir, "science_qa.jsonl"),
        (args.raw_dir, "qasper.jsonl"),
        (args.raw_dir, "scitldr.jsonl"),
        (args.raw_dir, "mmlu_cs.jsonl"),
        # Layer 3 Custom
        (args.custom_dir, "cs_fundamentals.jsonl"),
        (args.custom_dir, "ml_dl_concepts.jsonl"),
        (args.custom_dir, "advanced_ai_topics.jsonl"),
    ]

    phase1_samples = []
    print("Processing Phase 1 SFT files...")
    for directory, filename in tqdm(phase1_files):
        path = os.path.join(directory, filename)
        phase1_samples.extend(process_file(path))

    random.shuffle(phase1_samples)
    
    phase1_out = os.path.join(args.output_dir, "phase1_train.jsonl")
    with open(phase1_out, 'w', encoding='utf-8') as f:
        for s in phase1_samples:
            f.write(json.dumps(s) + '\n')
    print(f"Saved {len(phase1_samples)} samples to {phase1_out}")

    # -------------------------------------------------------------
    # PHASE 2 SFT: Layer 4 (Exams & Interviews)
    # -------------------------------------------------------------
    phase2_files = [
        (args.custom_dir, "gate_cs_prep.jsonl"),
        (args.custom_dir, "ml_interview_qa.jsonl"),
        (args.custom_dir, "dsa_interview.jsonl"),
        (args.custom_dir, "system_design_ml.jsonl"),
        (args.custom_dir, "viva_qa.jsonl"),
    ]

    phase2_samples = []
    print("Processing Phase 2 SFT files...")
    for directory, filename in tqdm(phase2_files):
        path = os.path.join(directory, filename)
        phase2_samples.extend(process_file(path))

    random.shuffle(phase2_samples)
    
    phase2_out = os.path.join(args.output_dir, "phase2_train.jsonl")
    with open(phase2_out, 'w', encoding='utf-8') as f:
        for s in phase2_samples:
            f.write(json.dumps(s) + '\n')
    print(f"Saved {len(phase2_samples)} samples to {phase2_out}")

    # -------------------------------------------------------------
    # PHASE 3 DPO: Layer 5 (Preferences)
    # -------------------------------------------------------------
    # DPO has a different schema: prompt, chosen, rejected.
    # We copy DPO pairs if they exist.
    dpo_custom_path = os.path.join(args.custom_dir, "dpo_preferences.jsonl")
    dpo_out = os.path.join(args.output_dir, "phase3_dpo.jsonl")
    
    dpo_count = 0
    if os.path.exists(dpo_custom_path):
        print("Processing Phase 3 DPO files...")
        with open(dpo_custom_path, 'r', encoding='utf-8') as infile, open(dpo_out, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "prompt" in data and "chosen" in data and "rejected" in data:
                            # Apply the system prompt to the prompt block
                            full_prompt = (
                                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                                f"<|im_start|>user\n{data['prompt']}<|im_end|>\n"
                            )
                            # Chosen/rejected should end with im_end
                            chosen_resp = f"{data['chosen']}<|im_end|>"
                            rejected_resp = f"{data['rejected']}<|im_end|>"
                            
                            outfile.write(json.dumps({
                                "prompt": full_prompt,
                                "chosen": chosen_resp,
                                "rejected": rejected_resp
                            }) + '\n')
                            dpo_count += 1
                    except Exception as e:
                        print(f"Error parsing DPO pair: {e}")
        print(f"Saved {dpo_count} DPO pairs to {dpo_out}")
    else:
        print("Warning: No custom DPO file found. Phase 3 dataset skipped.")

if __name__ == "__main__":
    main()
