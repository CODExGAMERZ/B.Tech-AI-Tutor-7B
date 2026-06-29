import os
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

def download_and_sample(dataset_name, split, sample_size, output_path, seed=42):
    print(f"Downloading {dataset_name} ({split})...")
    try:
        if dataset_name == "cais/mmlu":
            cs_subjects = ["college_computer_science", "computer_science", "high_school_computer_science"]
            datasets_list = []
            for subj in cs_subjects:
                try:
                    ds = load_dataset(dataset_name, subj, split=split)
                    datasets_list.append(ds)
                except Exception as e:
                    print(f"Warning: Could not load MMLU subject {subj}: {e}")
            from datasets import concatenate_datasets
            if datasets_list:
                dataset = concatenate_datasets(datasets_list)
            else:
                raise ValueError("No MMLU subjects loaded.")
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        total_samples = len(dataset)
        print(f"Loaded {total_samples} samples.")
        
        if sample_size and sample_size < total_samples:
            print(f"Sampling {sample_size} samples...")
            dataset = dataset.shuffle(seed=seed).select(range(sample_size))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f"Saving {os.path.basename(output_path)}"):
                f.write(json.dumps(item) + '\n')
                
        print(f"Successfully saved {len(dataset)} samples to {output_path}\n")
        return len(dataset)
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}\n")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Download and sample HuggingFace datasets for B.Tech AI Tutor")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save raw jsonl files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    
    datasets_to_download = [
        ("garage-bAInd/Open-Platypus", "train", None, "open_platypus.jsonl"),
        ("HuggingFaceH4/no_robots", "train", None, "no_robots.jsonl"),
        ("Open-Orca/SlimOrca", "train", 25000, "slim_orca.jsonl"),
        ("LDJnr/Capybara", "train", 10000, "capybara.jsonl"),
        
        ("nvidia/OpenMathInstruct-2", "train", 50000, "open_math_instruct.jsonl"),
        ("meta-math/MetaMathQA", "train", 25000, "meta_math_qa.jsonl"),
        ("TIGER-Lab/MathInstruct", "train", 15000, "math_instruct.jsonl"),
        ("microsoft/orca-math-word-problems-200k", "train", 10000, "orca_math.jsonl"),
        ("hendrycks/competition_math", "train", None, "competition_math.jsonl"),
        
        ("ise-uiuc/Magicoder-Evol-Instruct-110K", "train", 50000, "magicoder_evol.jsonl"),
        ("sahil2801/CodeAlpaca-20k", "train", None, "code_alpaca.jsonl"),
        ("glaiveai/glaive-code-assistant-v3", "train", 20000, "glaive_code.jsonl"),
        ("deepmind/code_contests", "train", 10000, "code_contests.jsonl"),
        ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "train", 10000, "self_oss_instruct.jsonl"),
        
        ("allenai/sciq", "train", None, "sciq.jsonl"),
        ("derek-thomas/ScienceQA", "train", None, "science_qa.jsonl"),
        ("allenai/qasper", "train", None, "qasper.jsonl"),
        ("allenai/scitldr", "train", None, "scitldr.jsonl"),
        ("cais/mmlu", "test", None, "mmlu_cs.jsonl") # MMLU split is test/validation
    ]
    
    summary = {}
    for name, split, size, filename in datasets_to_download:
        out_path = os.path.join(args.output_dir, filename)
        count = download_and_sample(name, split, size, out_path, seed=args.seed)
        summary[name] = count
        
    print("="*40)
    print("DOWNLOAD SUMMARY")
    print("="*40)
    for name, count in summary.items():
        print(f"{name}: {count} samples")
    print("="*40)

if __name__ == "__main__":
    main()
