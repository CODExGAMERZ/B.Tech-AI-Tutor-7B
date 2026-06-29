import os
import argparse
import json
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

def clean_and_normalize(text):
    return " ".join(text.lower().split())

def filter_dataset(input_file, output_file, threshold=0.8):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return 0, 0
        
    print(f"Filtering {input_file} -> {output_file}...")
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    total_seen = 0
    total_kept = 0
    malformed = 0
    too_short = 0
    duplicates = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, line in enumerate(tqdm(lines, desc="Filtering samples")):
            total_seen += 1
            if not line.strip():
                continue
                
            try:
                sample = json.loads(line)
            except Exception:
                malformed += 1
                continue
                
            if 'instruction' not in sample or 'output' not in sample:
                malformed += 1
                continue
                
            instruction = sample['instruction'].strip()
            output = sample['output'].strip()
            
            if len(instruction) < 20 or len(output) < 100:
                too_short += 1
                continue
                
            m = MinHash(num_perm=128)
            norm_text = clean_normalize(output)
            for word in norm_text.split():
                m.update(word.encode('utf-8'))
                
            dups = lsh.query(m)
            if dups:
                duplicates += 1
                continue
                
            lsh.insert(f"doc_{idx}", m)
            out_f.write(json.dumps({"instruction": instruction, "output": output}) + '\n')
            total_kept += 1
            
    print(f"Stats for {os.path.basename(input_file)}:")
    print(f"  Total seen: {total_seen}")
    print(f"  Total kept: {total_kept} ({total_kept/max(1, total_seen)*100:.1f}%)")
    print(f"  Malformed: {malformed}")
    print(f"  Too short: {too_short}")
    print(f"  Duplicates: {duplicates}")
    print("-" * 30)
    return total_seen, total_kept

def clean_normalize(text):
    return " ".join(text.lower().split())

def main():
    parser = argparse.ArgumentParser(description="Filter and deduplicate datasets using MinHash LSH")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save filtered jsonl files")
    parser.add_argument("--threshold", type=float, default=0.8, help="Jaccard threshold for deduplication")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]
    
    grand_total_seen = 0
    grand_total_kept = 0
    
    for filename in files:
        in_path = os.path.join(args.input_dir, filename)
        out_path = os.path.join(args.output_dir, filename)
        seen, kept = filter_dataset(in_path, out_path, threshold=args.threshold)
        grand_total_seen += seen
        grand_total_kept += kept
        
    print(f"\nGRAND TOTAL: Kept {grand_total_kept}/{grand_total_seen} ({grand_total_kept/max(1, grand_total_seen)*100:.1f}%)")

if __name__ == "__main__":
    main()
