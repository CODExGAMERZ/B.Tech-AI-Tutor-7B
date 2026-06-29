import os
import argparse
import json
import time
import random
from tqdm import tqdm
import google.generativeai as genai

TOPICS = [
    ("Operating Systems", "Process synchronization using Semaphores"),
    ("Operating Systems", "Deadlock detection and Banker's algorithm"),
    ("Operating Systems", "Page replacement algorithms (LRU vs Optimal)"),
    ("DBMS", "Normal forms: 3NF vs BCNF definition and decomposition"),
    ("DBMS", "Concurrency control and Two-Phase Locking (2PL)"),
    ("Computer Networks", "TCP Congestion Control (Slow Start, Congestion Avoidance)"),
    ("Computer Networks", "IP Subnetting and CIDR notation calculations"),
    ("TOC", "Designing DFA for regular languages"),
    ("TOC", "Undecidability and Halting Problem proof"),
    ("COA", "Cache memory mapping strategies"),
    ("Compiler Design", "LR Parsing table construction"),
    ("Machine Learning", "Bias-Variance Tradeoff derivation and explanation"),
    ("Machine Learning", "Support Vector Machines and Kernel Trick formulation"),
    ("Deep Learning", "Backpropagation calculus derivation"),
    ("Deep Learning", "Batch Normalization mechanics in training vs inference"),
    ("Generative AI & LLMs", "Scaled dot-product attention in Transformers"),
    ("MLOps", "Model drift monitoring and statistical test selection"),
    ("DSA", "Dijkstra's Shortest Path algorithm optimization"),
    ("DSA", "Binary Tree Level Order traversal implementation"),
    ("System Design", "Designing a high-throughput CTR prediction system"),
]

DPO_PROMPT_TEMPLATE = """You are a curriculum evaluator. Generate a DPO (Direct Preference Optimization) training triplet for the subject "{subject}" on the topic "{topic}".

Your output MUST be a JSON object with:
1. "prompt": A challenging question or problem on this topic.
2. "chosen": A top-tier, highly detailed response from an expert AI tutor. It must:
   - Provide clear, step-by-step reasoning.
   - Use LaTeX for mathematical formulas.
   - Include code with comprehensive comments (where applicable).
   - Be friendly, encouraging, and pedagogically sound.
3. "rejected": A poor-quality response to the same prompt. It should:
   - Be very brief or lazy.
   - Contain slight conceptual or mathematical errors.
   - Lack code comments or explanations.
   - Be formatting-deficient (e.g. no Markdown, no lists).
   - Use incorrect equations or skip crucial steps.

Output strictly as a valid JSON object:
{{"prompt": "...", "chosen": "...", "rejected": "..."}}"""

def clean_json_string(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs using Gemini API")
    parser.add_argument("--api_key", type=str, required=True, help="Google AI Studio API Key")
    parser.add_argument("--output_file", type=str, required=True, help="Output path for dpo_preferences.jsonl")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of preference pairs to generate")
    args = parser.parse_args()

    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"response_mime_type": "application/json"})

    existing_count = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_count += 1
        print(f"Loaded {existing_count} existing DPO pairs. Resuming...")

    if existing_count >= args.num_samples:
        print("Required DPO samples already generated.")
        return

    random.seed(42)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'a', encoding='utf-8') as f:
        pbar = tqdm(total=args.num_samples, initial=existing_count, desc="Generating DPO pairs")
        
        generated = existing_count
        while generated < args.num_samples:
            subject, topic = random.choice(TOPICS)
            prompt = DPO_PROMPT_TEMPLATE.format(subject=subject, topic=topic)
            
            try:
                response = model.generate_content(prompt)
                cleaned = clean_json_string(response.text)
                
                data = json.loads(cleaned)
                if "prompt" in data and "chosen" in data and "rejected" in data:
                    f.write(json.dumps(data) + '\n')
                    f.flush()
                    generated += 1
                    pbar.update(1)
                else:
                    print(f"\nWarning: Invalid schema. Response: {cleaned}")
            except Exception as e:
                print(f"\nError: {e}")
                time.sleep(2)
                
            time.sleep(0.5)
            
        pbar.close()
        print(f"Finished generating {generated} DPO pairs and saved to {args.output_file}")

if __name__ == "__main__":
    main()
