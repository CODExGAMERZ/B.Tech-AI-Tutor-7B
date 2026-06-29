import os
import argparse
import json
import time
import random
from tqdm import tqdm
import google.generativeai as genai

# Define all curriculum subjects and detailed topics
CURRICULUM = {
    "cs_fundamentals": {
        "Operating Systems": [
            "Process states and state transitions", "CPU scheduling algorithms (FCFS, SJF, SRTF, Round Robin, Priority)",
            "Process synchronization, critical section problem, Peterson's solution", "Semaphores, Mutexes, Monitors",
            "Classical synchronization problems (Bounded Buffer, Readers-Writers, Dining Philosophers)",
            "Deadlocks: characterization, prevention, avoidance (Banker's algorithm), detection, recovery",
            "Memory management: logical vs physical address space, swapping, contiguous allocation",
            "Paging: page table structure, TLB, multi-level paging", "Segmentation: concepts, hardware, fragmentation",
            "Virtual memory: demand paging, page faults, page replacement algorithms (FIFO, Optimal, LRU, LFU)",
            "Thrashing: causes, working set model, page fault frequency", "File systems: interface, mounting, sharing, protection",
            "Directory structure and file system implementation: allocation methods, free space management",
            "Disk structure, scheduling (FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK), RAID levels",
            "I/O hardware, application I/O interface, kernel I/O subsystem", "Protection and security: goals, domain, access matrix, threats"
        ],
        "DBMS": [
            "Three-schema architecture and data independence", "Entity-Relationship (ER) model: entities, attributes, relationships, keys",
            "Relational model constraints: domain, key, referential integrity", "Relational algebra: select, project, join, set operations",
            "SQL: DDL (Data Definition Language) commands and syntax", "SQL: DML (Data Manipulation Language) queries, joins, subqueries",
            "SQL: Aggregate functions, GROUP BY, HAVING, views", "Functional dependencies and closure computation",
            "Normalization: 1NF, 2NF, 3NF definition and decomposition", "Normalization: Boyce-Codd Normal Form (BCNF) and dependency preservation",
            "Normalization: 4NF, 5NF, and multi-valued dependencies", "Transaction concepts and ACID properties",
            "Concurrency control: Lock-based protocols (2PL, Strict 2PL)", "Concurrency control: Timestamp-based, validation-based protocols",
            "Deadlock handling in transactions: detection, prevention, wait-die, wound-wait",
            "Database recovery techniques: log-based recovery, deferred/immediate update, checkpoints",
            "Indexing: B-tree and B+ tree indexing structures and operations", "Hashing: static and dynamic hashing, collision resolution"
        ],
        "Computer Networks": [
            "OSI reference model vs TCP/IP protocol suite", "Physical layer: transmission media, modulation, multiplexing",
            "Data link layer: framing, error detection (CRC, checksum) and correction",
            "Flow control: Stop-and-Wait, Go-Back-N, Selective Repeat protocols",
            "Medium Access Control: ALOHA, CSMA/CD, CSMA/CA, token passing",
            "Ethernet standards, switching, VLANs, ARP protocol", "Network layer: IPv4 addressing, subnetting, CIDR notation",
            "IP routing algorithms: distance vector (RIP), link state (OSPF), path vector (BGP)",
            "ICMP, DHCP, NAT protocols and operational workflows", "IPv6 structure, address representation, transition mechanisms",
            "Transport layer: port numbers, UDP connectionless service structure",
            "TCP: connection establishment (3-way handshake), termination, segment structure",
            "TCP flow control (sliding window) and congestion control (slow start, congestion avoidance)",
            "Application layer: DNS architecture, resolution process, record types",
            "HTTP and HTTPS: request/response, headers, cookies, SSL/TLS handshake",
            "FTP, SMTP, POP3, IMAP mail protocols and operations", "Network security: cryptography, firewalls, IPSec, VPNs"
        ],
        "TOC": [
            "Deterministic Finite Automata (DFA) design and formal definition",
            "Non-deterministic Finite Automata (NFA) design, equivalence to DFA",
            "Regular expressions: syntax, equivalence to finite automata, closure properties",
            "Pumping Lemma for regular languages and proving non-regularity", "Minimization of finite automata (Myhill-Nerode theorem)",
            "Context-Free Grammars (CFG): derivation trees, ambiguity, Chomsky Normal Form (CNF)",
            "Pushdown Automata (PDA): design, equivalence to CFGs, deterministic PDAs",
            "Pumping Lemma for Context-Free Languages and proving non-context-free",
            "Turing Machines: design, variants, transition diagrams, Church-Turing thesis",
            "Decidability: recursive and recursively enumerable languages, halting problem",
            "Undecidable problems: Post Correspondence Problem, Rice's theorem",
            "Complexity classes: P, NP, NP-complete, NP-hard, Cook-Levin theorem"
        ],
        "COA": [
            "Computer registers, bus system, instruction cycle", "Instruction formats: address instructions, addressing modes",
            "Control unit design: hardwired vs microprogrammed control",
            "Arithmetic pipeline: adder, multiplier, instruction pipeline hazards",
            "Memory hierarchy: cache memory mapping (direct, associative, set-associative)",
            "Cache coherence protocols (MESI), write-through vs write-back",
            "Virtual memory: page tables, address translation, TLB interaction",
            "Input-Output organization: programmed I/O, interrupt-initiated I/O, DMA",
            "Pipelining hazards: structural, data, control hazards and mitigation",
            "Reduced Instruction Set Computer (RISC) vs Complex Instruction Set Computer (CISC)",
            "Multiprocessor organizations: symmetric multiprocessors, NUMA, clusters"
        ],
        "Compiler Design": [
            "Phases of a compiler: lexical, syntax, semantic analysis, synthesis",
            "Lexical analyzer design: token recognition, transition diagrams, Lex tool",
            "Context-free grammars and parsing: top-down parsing (LL(1), recursive descent)",
            "Bottom-up parsing: shift-reduce, operator precedence parsing",
            "LR parsing: LR(0), SLR(1), canonical LR(1), LALR(1) parsing tables",
            "Syntax-directed translation: synthesized and inherited attributes, dependency graphs",
            "Intermediate code representation: three-address code, quadruples, triples",
            "Runtime environments: activation records, storage allocation, stack allocation",
            "Code optimization: local, loop, global optimization, data flow analysis",
            "Code generation: register allocation, instruction selection, DAG representation"
        ]
    },
    "ml_dl_concepts": {
        "Machine Learning": [
            "Supervised learning paradigms: regression vs classification", "Linear regression: formulation, loss, gradient descent, normal equation",
            "Regularization: Ridge (L2), Lasso (L1), Elastic Net optimization",
            "Logistic regression: sigmoid function, cross-entropy loss, decision boundary",
            "Support Vector Machines (SVM): maximum margin, dual formulation, kernel trick",
            "Decision Trees: entropy, information gain, Gini index, ID3, C4.5, CART",
            "Ensemble methods: Bagging, Random Forests, out-of-bag error",
            "Ensemble methods: Boosting, AdaBoost, Gradient Boosting, XGBoost, LightGBM",
            "K-Nearest Neighbors (KNN): distance metrics, curse of dimensionality",
            "Naive Bayes classifier: conditional probability, Laplace smoothing",
            "Unsupervised learning: K-Means clustering, elbow method, silhouette analysis",
            "Hierarchical clustering: agglomerative vs divisive, linkage criteria",
            "Density-based clustering: DBSCAN parameters, core/border/noise points",
            "Dimensionality reduction: Principal Component Analysis (PCA) derivation, explained variance",
            "Dimensionality reduction: t-SNE, UMAP non-linear projection concepts",
            "Model evaluation: bias-variance tradeoff, cross-validation techniques",
            "Evaluation metrics: precision, recall, F1-score, ROC-AUC, confusion matrix",
            "Feature engineering: scaling, encoding, imputation, selection techniques"
        ],
        "Deep Learning": [
            "Artificial neurons, perceptron convergence theorem, multi-layer perceptron (MLP)",
            "Backpropagation derivation: chain rule, weight update formulas",
            "Activation functions: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, GELU, Softmax",
            "Optimization algorithms: SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam, AdamW",
            "Vanishing and exploding gradients: causes, solutions (clipping, initialization)",
            "Weight initialization techniques: Xavier/Glorot, He/Kaiming initialization",
            "Overfitting prevention: L1/L2 weight decay, dropout, early stopping",
            "Batch Normalization, Layer Normalization: math, operations, training vs inference",
            "Convolutional Neural Networks (CNN): convolution, pooling, padding, stride operations",
            "CNN architectures: LeNet, AlexNet, VGG, ResNet skip connections, Inception modules",
            "Recurrent Neural Networks (RNN): structure, backpropagation through time (BPTT)",
            "LSTM and GRU: gating mechanisms, cell state math, solving long-term dependencies",
            "Autoencoders: architecture, reconstruction loss, undercomplete/denoising variants",
            "Variational Autoencoders (VAE): latent space, KL divergence, reparameterization trick",
            "Generative Adversarial Networks (GAN): minimax game, generator/discriminator loss, training challenges"
        ]
    },
    "advanced_ai_topics": {
        "Generative AI & LLMs": [
            "Attention mechanism: additive vs multiplicative, self-attention math",
            "Transformer architecture: scaled dot-product attention, multi-head attention, positional encoding",
            "Encoder-Decoder models: BERT pre-training (MLM, NSP), GPT pre-training (causal LM)",
            "T5 architecture, sequence-to-sequence unified framework", "Byte Pair Encoding (BPE), WordPiece, SentencePiece tokenization",
            "Decoders: Greedy decoding, Beam search, temperature scaling, Top-K, Top-p (nucleus) sampling",
            "Parameter-Efficient Fine-Tuning (PEFT): LoRA (Low-Rank Adaptation) mathematical formulation",
            "PEFT variants: QLoRA, Prefix Tuning, Prompt Tuning, AdaLoRA",
            "Retrieval-Augmented Generation (RAG): architecture, vector databases, chunking strategies",
            "Agentic workflows: ReAct pattern, tool use, memory management in agents",
            "Mixture of Experts (MoE): routing gate mechanism, sparse activation",
            "Vision Transformers (ViT): patch projection, class token, self-attention on patches",
            "Diffusion Models: forward diffusion, reverse denoising process, classifier-free guidance"
        ],
        "MLOps": [
            "MLOps lifecycle, data validation, schema detection", "Data version control: DVC architecture, integration with Git",
            "Experiment tracking: MLflow, Weights & Biases configuration and tracking",
            "Feature stores: Feast architecture, online vs offline feature serving",
            "Containerization: Dockerizing ML models, multi-stage builds",
            "Model deployment: FastAPI, Triton Inference Server, TorchServe",
            "Orchestration: Airflow, Kubeflow pipelines for ML training automation",
            "CI/CD for ML: GitHub Actions, automated testing of models",
            "Model monitoring: concept drift, data drift detection, statistical tests (KS test)",
            "Scalable model serving: Kubernetes, Knative, autoscaling ML endpoints"
        ]
    },
    "gate_prep": {
        "GATE CS": [
            "GATE: Programming and Data Structures - Pointers, Recursion, Trees, Graphs",
            "GATE: Algorithms - Greedy, Dynamic Programming, Complexity bounds, Sorting",
            "GATE: Operating Systems - Semaphores, CPU Scheduling, Deadlock detection, Paging math",
            "GATE: DBMS - Normalization (BCNF, 3NF key detection), Relational Algebra queries, Transactions serializability",
            "GATE: Computer Networks - IP addressing, Subnetting, TCP Congestion control math, Sliding window efficiency",
            "GATE: TOC - Minimization of DFA, Decidability closure properties, Context-free grammar parsing",
            "GATE: Compiler Design - LL(1) parse table, First and Follow sets, LR(0)/SLR(1) state transitions",
            "GATE: COA - Cache mapping (set size, index bits), Pipeline speedup, Addressing modes math",
            "GATE: Discrete Mathematics - Propositional logic, Set theory cardinality, Graph theory properties (coloring, planar graphs)",
            "GATE: Engineering Mathematics - Linear algebra eigenvalues, Probability Bayes' theorem, Calculus limits"
        ]
    },
    "ml_interview": {
        "ML DL Interview": [
            "Explain bias-variance tradeoff mathematically and conceptually", "How does SVM handle non-linear data? Explain the dual formulation",
            "Contrast Random Forest vs XGBoost in terms of variance, bias, parallelization",
            "What is the mathematical formulation of L1 vs L2 regularization?",
            "Explain backpropagation from scratch with calculus equations",
            "Why do we need activation functions? What happens if we use linear activations?",
            "Explain Batch Normalization math and how it behaves differently during training vs inference",
            "What is the vanishing gradient problem? Explain 3 methods to mitigate it",
            "Explain self-attention math in Transformers. Why is the scale factor 1/sqrt(d_k) used?",
            "What is the difference between causal attention and bidirectional attention?",
            "Explain the reparameterization trick in Variational Autoencoders",
            "What is Wasserstein GAN? How does it improve over standard GAN training?"
        ]
    },
    "dsa_interview": {
        "DSA Interview": [
            "LeetCode: Arrays and Hashing - Two Sum, Group Anagrams, Top K Frequent Elements",
            "LeetCode: Two Pointers - Valid Palindrome, 3Sum, Container With Most Water",
            "LeetCode: Sliding Window - Longest Substring Without Repeating Characters, Minimum Window Substring",
            "LeetCode: Stack - Valid Parentheses, Min Stack, Daily Temperatures",
            "LeetCode: Binary Search - Search in Rotated Sorted Array, Find Minimum in Rotated Sorted Array",
            "LeetCode: LinkedList - Reverse Linked List, Merge Two Sorted Lists, Reorder List",
            "LeetCode: Trees - Maximum Depth of Binary Tree, Invert Binary Tree, Binary Tree Level Order Traversal",
            "LeetCode: Graphs - Number of Islands, Clone Graph, Course Schedule, Pacifc Atlantic Water Flow",
            "LeetCode: Heap / Priority Queue - Kth Largest Element in an Array, Merge k Sorted Lists",
            "LeetCode: Backtracking - Subsets, Permutations, Combination Sum, Word Search",
            "LeetCode: Dynamic Programming - Climbing Stairs, Coin Change, Longest Common Subsequence, Edit Distance",
            "LeetCode: Greedy - Jump Game, Gas Station, Hand of Straights"
        ]
    },
    "system_design_ml": {
        "ML System Design": [
            "Design a Real-time Recommendation System (e.g., Netflix, YouTube)",
            "Design a Search Relevance and Ranking System (e.g., Google Search, Amazon)",
            "Design an Ad Click-Through Rate (CTR) Prediction Pipeline",
            "Design a Large-Scale Image/Video Search Engine (Visual Search)",
            "Design a Real-time Fraud Detection System for Financial Transactions",
            "Design a Conversational AI / Virtual Assistant System (RAG based)",
            "Design an Autonomous Driving Perception Pipeline (Sensor Fusion)",
            "Design an ML Pipeline for Ride-Sharing ETA Prediction (e.g., Uber)",
            "Design a Large-Scale Document Classification and Search System",
            "Design an ML-based Anomaly Detection System for IT Infrastructure"
        ]
    },
    "viva_qa": {
        "Viva Questions": [
            "DBMS Viva: Explain transaction ACID properties and index types",
            "OS Viva: Contrast Paging vs Segmentation, and explain Virtual Memory",
            "Computer Networks Viva: Explain TCP 3-way handshake and OSI layers",
            "ML Viva: Explain SVM margins, Kernels, and Bias-Variance tradeoff",
            "DL Viva: Explain CNN pooling, strides, Backprop, and optimizers",
            "TOC Viva: Define DFA, NFA, Context-Free Grammars, and Turing Machines",
            "Algorithms Viva: Explain Dynamic Programming vs Greedy approaches with examples",
            "Software Engineering Viva: Explain Agile methodology and Design Patterns"
        ]
    }
}

PROMPTS = {
    "cs_fundamentals": """You are an elite B.Tech Computer Science professor. Generate a detailed, highly accurate instructional Q&A pair on "{subject}" for the topic "{topic}".

Requirements:
- Question: Academic exam style or GATE level, testing deep understanding.
- Answer: Comprehensive (200-500 words) containing:
  1. Conceptual Explanation (clear and precise)
  2. Mathematical formulation or algorithm representation (where applicable, using LaTeX for equations)
  3. Concrete worked example or code snippet
  4. Common mistakes, edge cases, or misconceptions to be aware of
  5. A follow-up question for self-reflection

Output strictly as a valid JSON object:
{{"instruction": "<question>", "output": "<detailed answer>"}}""",

    "ml_dl_concepts": """You are an ML researcher and B.Tech AI/ML professor. Create a detailed instructional Q&A about "{concept}" in "{area}".

The answer MUST include:
1. Intuitive Explanation (clear analogy, then technical breakdown)
2. Mathematical Formulation (detailed equations in LaTeX format)
3. Python code snippet (using PyTorch, NumPy, or scikit-learn with thorough comments)
4. Real-world application example
5. Common interview/exam follow-up question

Output strictly as a valid JSON object:
{{"instruction": "...", "output": "..."}}""",

    "advanced_ai_topics": """You are an AI Architect. Create a technical instruction Q&A on "{concept}" in "{area}".

Include:
1. Architectural block diagrams (represented in text or markdown tables) or deep structural explanation
2. Mathematical implementation details (using LaTeX)
3. Production-level code snippet (e.g., PyTorch module or MLOps config)
4. Trade-offs (latency vs accuracy, memory vs compute)
5. Production challenges and mitigations

Output strictly as a valid JSON object:
{{"instruction": "...", "output": "..."}}""",

    "gate_prep": """Generate a GATE Computer Science exam-style question for {subject} on the topic {topic}.

Include:
- Question (MCQ format with 4 options: A, B, C, D; or Numerical Answer Type NAT)
- Correct Answer
- Step-by-step mathematical/logical solution showing exact working
- Relevant GATE concepts and formulas to memorize
- Time-saving shortcuts or tips for this type of problem

Output strictly as a valid JSON object:
{{"instruction": "...", "output": "..."}}""",

    "ml_interview": """Generate an advanced Machine Learning/Deep Learning interview Q&A on the topic: "{topic}".

The answer should represent a top-tier candidate's response:
- Structure: Direct answer → Math/Theory → Trade-offs → Edge Cases → Practical Experience/System considerations
- Include clear LaTeX math where appropriate
- Discuss production deployment implications of this topic

Output strictly as a valid JSON object:
{{"instruction": "...", "output": "..."}}""",

    "dsa_interview": """Generate an elite Software Engineering coding interview question based on "{topic}".

Include:
- Problem Description (LeetCode style, clear inputs/outputs and constraints)
- Example test cases (at least 2)
- Optimal Solution in Python (clean, well-commented, using correct type hints)
- Complexity Analysis: Step-by-step derivation of Time and Space complexity
- Alternative approaches (e.g., Brute force vs DP vs Greedy) and why the proposed one is optimal

Output strictly as a valid JSON object:
{{"instruction": "...", "output": "..."}}""",

    "system_design_ml": """You are a Principal ML Systems Engineer. Generate an ML System Design interview question and answer for: "{topic}".

The response should cover:
1. Requirements Elaboration (Functional & Non-Functional)
2. Data Pipeline & Feature Engineering (Offline & Online)
3. Model Architecture & Selection
4. Training Pipeline & Evaluation (Offline & Online metrics)
5. Serving & Scaling Infrastructure (MLOps, Latency SLAs, Caching, Fallbacks)
6. Monitoring & Continuous Learning (Feedback loops, Drift detection)

Output strictly as a valid JSON object:
{{"instruction": "...", "output": "..."}}""",

    "viva_qa": """Generate a university viva (oral exam) Q&A for {subject} on: "{topic}".

Include:
- Professor's opening question
- Recommended student answer (concise, high-impact, hitting key keywords)
- Likely follow-up grilling questions by the external examiner (at least 2) and how to handle them
- Key jargon/terminology to definitely mention

Output strictly as a valid JSON object:
{{"instruction": "...", "output": "..."}}"""
}

def clean_json_string(text):
    """Clean markdown markers or trailing text that the model might generate."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate custom synthetic datasets using Google Gemini API")
    parser.add_argument("--api_key", type=str, required=True, help="Google AI Studio API Key")
    parser.add_argument("--dataset_type", type=str, required=True, 
                        choices=list(CURRICULUM.keys()), help="Dataset category to generate")
    parser.add_argument("--output_file", type=str, required=True, help="Output path for the jsonl file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()

    # Configure Gemini API
    genai.configure(api_key=args.api_key)
    # Using gemini-2.0-flash as the fast, cheap, high-quality model
    model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"response_mime_type": "application/json"})

    # Setup curriculum items
    category_data = CURRICULUM[args.dataset_type]
    all_pairs = []
    for subject, topics in category_data.items():
        for topic in topics:
            all_pairs.append((subject, topic))

    # Read existing progress if any
    existing_samples = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        existing_samples.append(json.loads(line))
                    except:
                        pass
        print(f"Loaded {len(existing_samples)} existing samples. Resuming generation...")

    start_idx = len(existing_samples)
    if start_idx >= args.num_samples:
        print("Required samples already generated.")
        return

    # Randomly shuffle items to generate broad coverage
    random.seed(42)
    random.shuffle(all_pairs)

    # Open output file in append mode
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'a', encoding='utf-8') as f:
        pbar = tqdm(total=args.num_samples, initial=start_idx, desc=f"Generating {args.dataset_type}")
        
        generated_count = start_idx
        while generated_count < args.num_samples:
            # Select subject and topic
            subject, topic = all_pairs[generated_count % len(all_pairs)]
            
            prompt_template = PROMPTS[args.dataset_type]
            if args.dataset_type in ["cs_fundamentals", "gate_prep", "viva_qa"]:
                prompt = prompt_template.format(subject=subject, topic=topic)
            elif args.dataset_type in ["ml_dl_concepts", "advanced_ai_topics"]:
                prompt = prompt_template.format(concept=topic, area=subject)
            else:
                prompt = prompt_template.format(topic=topic)
                
            try:
                response = model.generate_content(prompt)
                cleaned_text = clean_json_string(response.text)
                
                # Verify JSON structure
                data = json.loads(cleaned_text)
                if "instruction" in data and "output" in data:
                    f.write(json.dumps(data) + '\n')
                    f.flush()
                    generated_count += 1
                    pbar.update(1)
                else:
                    print(f"\nWarning: Missing instruction or output key. Response: {cleaned_text}")
            except Exception as e:
                print(f"\nError generating sample {generated_count}: {e}")
                time.sleep(2)  # Backoff on error
                
            time.sleep(0.5)  # Rate limit safety
            
        pbar.close()
        print(f"Finished generating {generated_count} samples and saved to {args.output_file}")

if __name__ == "__main__":
    main()
