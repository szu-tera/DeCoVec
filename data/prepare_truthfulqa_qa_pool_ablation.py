"""
TruthfulQA dataset loading and preprocessing (QA format) - Pool Ablation Version
Uses few-shot prompt format, suitable for open-ended generation and multiple-choice tasks

Data description:
- Uses TruthfulQA.csv data
- Uses Best Answer as training target
- Uses few-shot prompt format (according to paper Table 10)
- Training: Learn to generate Best Answer
- Evaluation: Can be used for open-ended generation and multiple-choice tasks
"""
import json
import os
import random
import csv
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config


def load_truthfulqa_from_csv(csv_file: str) -> List[Dict]:
    """
    Load data from TruthfulQA.csv
    
    Args:
        csv_file: CSV file path
    
    Returns:
        List of data examples
    """
    print(f"Loading TruthfulQA dataset from local file: {csv_file}")
    
    data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "type": row["Type"],
                    "category": row["Category"],
                    "question": row["Question"],
                    "best_answer": row["Best Answer"],
                    "correct_answers": row["Correct Answers"],
                    "incorrect_answers": row["Incorrect Answers"],
                    "source": row["Source"]
                })
        print(f"✓ Dataset loaded successfully: {len(data)} examples")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return []
    
    return data


def format_qa_example(example: Dict, mc_data: Dict = None) -> Dict:
    """
    Format QA sample
    
    Note: Prompt templates (instruction, few_shot_examples) have been moved to data/truthfulqa/prompt.json
    Data only contains pure data fields
    
    Key design:
    1. Use Best Answer as training target
    2. Get MC1/MC2 option sets from mc_task.json
    
    Args:
        example: Raw data sample (from CSV)
        mc_data: MC task data (from mc_task.json), includes mc1/mc2 options
    
    Returns:
        Formatted sample
    """
    question = example["question"]
    best_answer = example["best_answer"]
    
    # Validate data integrity
    if not question or not best_answer:
        return None
    
    # Get MC1/MC2 options from mc_data (if provided)
    mc1_choices = []
    mc1_correct_idx = None
    mc2_choices = []
    mc2_correct_choices = []
    
    if mc_data is not None:
        # mc1_targets: {choice: label, ...} where label=1 means correct, 0 means incorrect
        mc1_targets = mc_data.get("mc1_targets", {})
        if mc1_targets:
            mc1_choices = list(mc1_targets.keys())
            # Find the index of correct answer (label=1)
            for idx, (choice, label) in enumerate(mc1_targets.items()):
                if label == 1:
                    mc1_correct_idx = idx
                    break
        
        # mc2_targets: {choice: label, ...} may have multiple correct answers (label=1)
        mc2_targets = mc_data.get("mc2_targets", {})
        if mc2_targets:
            mc2_choices = list(mc2_targets.keys())
            # Collect all correct answers
            mc2_correct_choices = [choice for choice, label in mc2_targets.items() if label == 1]
    
    # Return data structure (only pure data fields, removed instruction and few_shot_examples)
    return {
        # ===== Core Fields =====
        "question": question,                          # Original question text
        "best_answer": best_answer,                    # Best answer (training target)
        
        # ===== MC1 Evaluation Fields =====
        "mc1_choices": mc1_choices,              # MC1 option list
        "mc1_correct_idx": mc1_correct_idx,      # MC1 correct answer index
        
        # ===== MC2/MC3 Evaluation Fields =====
        "mc2_choices": mc2_choices,              # MC2 option list (all possible answers)
        "mc2_correct_choices": mc2_correct_choices,  # MC2 correct answer list
        
        # ===== Backup Fields (from CSV) =====
        "correct_answers": example["correct_answers"],      # All correct answers (CSV format)
        "incorrect_answers": example["incorrect_answers"],  # Incorrect answers (CSV format)
        
        # ===== Metadata =====
        "type": example["type"],
        "category": example["category"],
        "source": example["source"]
    }


def prepare_truthfulqa_qa():
    """
    Main function: TruthfulQA pool size ablation version
    - Total 790 samples
    - Randomly take 290 as test set
    - Remaining 500 as example selection pool (calibration)
    """
    config = get_config()
    
    # 1. Load dataset from local CSV (using files in original_dataset)
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_dataset", "TruthfulQA-main", "TruthfulQA.csv")
    raw_data = load_truthfulqa_from_csv(csv_file)
    
    if not raw_data:
        print("✗ Data loading failed")
        return
    
    # 2. Load mc_task.json to get MC1/MC2 options (using files in original_dataset)
    mc_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_dataset", "TruthfulQA-main", "data", "mc_task.json")
    mc_data_dict = {}
    
    if os.path.exists(mc_file):
        print(f"Loading MC task data: {mc_file}")
        with open(mc_file, 'r', encoding='utf-8') as f:
            mc_data_list = json.load(f)
        
        # Build question -> mc_data mapping
        for mc_item in mc_data_list:
            question = mc_item.get("question", "")
            mc_data_dict[question] = mc_item
        
        print(f"✓ MC task data loaded successfully: {len(mc_data_dict)} examples")
    else:
        print(f"⚠️  mc_task.json not found, MC1/MC2 options will be empty")
    
    # 3. Process QA data
    print("\n" + "="*60)
    print("Processing TruthfulQA QA data")
    print("="*60)
    
    qa_data = []
    skipped = 0
    
    for example in tqdm(raw_data, desc="Formatting"):
        # Find corresponding MC data
        question = example["question"]
        mc_data = mc_data_dict.get(question, None)
        
        formatted = format_qa_example(example, mc_data)
        if formatted is not None:
            qa_data.append(formatted)
        else:
            skipped += 1
    
    print(f"✓ Successfully processed {len(qa_data)} examples")
    if skipped > 0:
        print(f"  Skipped {skipped} incomplete examples")
    
    if len(qa_data) != 790:
        raise ValueError(f"TruthfulQA total samples should be 790, but got {len(qa_data)}. Please check original CSV.")

    rng = random.Random(config.seed)
    qa_shuffled = qa_data.copy()
    rng.shuffle(qa_shuffled)

    test_size = 290
    pool_size = 500

    test_data = qa_shuffled[:test_size]
    pool_data = qa_shuffled[test_size:test_size + pool_size]

    if len(pool_data) != pool_size:
        raise ValueError(f"TruthfulQA example pool should be {pool_size}, got {len(pool_data)}.")

    # Save to separate directory to avoid overwriting original data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truthfulqa_pool_ablation")
    os.makedirs(data_dir, exist_ok=True)

    test_path = os.path.join(data_dir, "truthfulqa_pool_ablation.json")
    cal_path = os.path.join(data_dir, "truthfulqa_pool_ablation_calibration.json")

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    with open(cal_path, 'w', encoding='utf-8') as f:
        json.dump(pool_data, f, ensure_ascii=False, indent=2)

    print("\nData statistics:")
    print(f"  - Test set: {len(test_data)}")
    print(f"  - Example pool/calibration set: {len(pool_data)}")
    type_counts = {}
    for item in pool_data:
        type_counts[item['type']] = type_counts.get(item['type'], 0) + 1
    print(f"  - Example pool type distribution (Top3): " + ", ".join(
        f"{t}:{c}" for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    ))

    print("\nSample examples (first 2 from test set):")
    for i, item in enumerate(test_data[:2], start=1):
        print(f"  Sample {i}: {item['question'][:80]}{'...' if len(item['question'])>80 else ''}")
        print(f"    Best answer: {item['best_answer'][:120]}{'...' if len(item['best_answer'])>120 else ''}")

    print("\n" + "="*60)
    print("✓ TruthfulQA pool size ablation data generated")
    print("  Test set ->", test_path)
    print("  Example pool ->", cal_path)
    print("="*60)


def load_data(split: str = "all") -> Dict[str, List[Dict]]:
    """Load TruthfulQA pool size ablation data.

    Args:
        split: "all", "test", "calibration"
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truthfulqa_pool_ablation")
    test_file = os.path.join(data_dir, "truthfulqa_pool_ablation.json")
    cal_file = os.path.join(data_dir, "truthfulqa_pool_ablation_calibration.json")

    result: Dict[str, List[Dict]] = {}
    if split in {"all", "test"}:
        with open(test_file, 'r', encoding='utf-8') as f:
            result["test"] = json.load(f)
    if split in {"all", "calibration"}:
        with open(cal_file, 'r', encoding='utf-8') as f:
            result["calibration"] = json.load(f)
    return result


if __name__ == "__main__":
    prepare_truthfulqa_qa()

