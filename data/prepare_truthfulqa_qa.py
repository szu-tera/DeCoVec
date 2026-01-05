"""
TruthfulQA dataset loading and preprocessing (QA format)
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
        Data list
    """
    print(f"Loading TruthfulQA dataset from local: {csv_file}")
    
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
        print(f"✓ Dataset loaded successfully: {len(data)} samples")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return []
    
    return data


def format_qa_example(example: Dict, mc_data: Dict = None) -> Dict:
    """
    Format QA sample
    
    Note: Prompt templates (instruction, few_shot_examples) moved to data/truthfulqa/prompt.json
    Data only contains pure data fields
    
    Key design:
    1. Use Best Answer as training target
    2. Get MC1/MC2 choice sets from mc_task.json
    
    Args:
        example: Raw data sample (from CSV)
        mc_data: MC task data (from mc_task.json), contains mc1/mc2 choices
    
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


def split_train_calibration(data: List[Dict], calibration_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset according to paper method
    
    Args:
        data: Complete dataset (Dtask)
        calibration_ratio: Calibration set ratio (default 10%)
        seed: Random seed
    
    Returns:
        (full_data, train_data, calibration_data)
        - full_data: Complete dataset for final testing
        - train_data: Reserved for future fine-tuning experiments (not used in current paper)
        - calibration_data: Calibration set (Dcalib), for computing global μ̄
    """
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Calculate split point
    total_size = len(data_copy)
    calibration_size = int(total_size * calibration_ratio)
    
    # Split: Dcalib (10%) + Dtrain (90%)
    calibration = data_copy[:calibration_size]
    train = data_copy[calibration_size:]
    
    print(f"✓ Data split complete:")
    print(f"  - Full dataset: {total_size} samples (for final testing)")
    print(f"  - Training set (Dtrain): {len(train)} samples ({len(train)/total_size*100:.1f}%)")
    print(f"  - Calibration set (Dcalib): {len(calibration)} samples ({len(calibration)/total_size*100:.1f}%)")
    
    return data_copy, train, calibration


def prepare_truthfulqa_qa():
    """
    Main function: Load and preprocess TruthfulQA data (QA format)
    Split data according to standard practice:
    - Dtask (full set) → for final testing
    - Dcalib (10%) → for computing global μ̄
    - Dtrain (90%) → reserved for future experiments (not used in current paper)
    """
    config = get_config()
    
    # 1. Load dataset from local CSV (using files in original_dataset)
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_dataset", "TruthfulQA-main", "TruthfulQA.csv")
    raw_data = load_truthfulqa_from_csv(csv_file)
    
    if not raw_data:
        print("✗ Data loading failed")
        return
    
    # 2. Load mc_task.json to get MC1/MC2 choices (using files in original_dataset)
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
        
        print(f"✓ MC task data loaded successfully: {len(mc_data_dict)} samples")
    else:
        print(f"⚠️  mc_task.json not found, MC1/MC2 choices will be empty")
    
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
    
    print(f"✓ Successfully processed {len(qa_data)} samples")
    if skipped > 0:
        print(f"  Skipped {skipped} incomplete samples")
    
    # Print data statistics
    print("\nData statistics:")
    print(f"  - Total samples: {len(qa_data)}")
    print(f"  - Type distribution:")
    type_counts = {}
    for item in qa_data:
        type_counts[item['type']] = type_counts.get(item['type'], 0) + 1
    for t, count in type_counts.items():
        print(f"    * {t}: {count}")
    
    # Calculate average lengths
    avg_question_len = sum(len(d['question']) for d in qa_data) / len(qa_data)
    avg_answer_len = sum(len(d['best_answer']) for d in qa_data) / len(qa_data)
    
    print(f"\nLength statistics:")
    print(f"  - Average question length: {avg_question_len:.1f} chars")
    print(f"  - Average answer length: {avg_answer_len:.1f} chars")
    
    # MC1/MC2 choice statistics
    mc1_choice_counts = [len(d['mc1_choices']) for d in qa_data if d['mc1_choices']]
    mc2_choice_counts = [len(d['mc2_choices']) for d in qa_data if d['mc2_choices']]
    mc2_correct_counts = [len(d['mc2_correct_choices']) for d in qa_data if d['mc2_correct_choices']]
    
    if mc1_choice_counts:
        print(f"\nMC choice statistics:")
        print(f"  - MC1 average choices: {sum(mc1_choice_counts) / len(mc1_choice_counts):.1f}")
        print(f"  - MC2 average choices: {sum(mc2_choice_counts) / len(mc2_choice_counts):.1f}")
        print(f"  - MC2 average correct answers: {sum(mc2_correct_counts) / len(mc2_correct_counts):.1f}")
        print(f"  - Samples with MC data: {len(mc1_choice_counts)}/{len(qa_data)}")
    
    # Print first 2 sample examples
    print("\nSample examples (first 2):")
    for i in range(min(2, len(qa_data))):
        item = qa_data[i]
        print(f"\n  Sample {i+1}:")
        print(f"  Question: {item['question']}")
        print(f"  Best answer: {item['best_answer']}")
        print(f"  Type: {item['type']}")
        print(f"  Category: {item['category']}")
    
    # 3. Split according to paper method: full, train, calibration
    qa_full, qa_train, qa_calibration = split_train_calibration(
        qa_data,
        config.data.calibration_split,  # Default 0.1 (10%)
        config.seed
    )
    
    # 4. Save complete dataset (for final testing)
    # Use new filename to avoid conflict with original MC data
    # Save to data/truthfulqa/ subdirectory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truthfulqa")
    os.makedirs(data_dir, exist_ok=True)
    qa_file = os.path.join(data_dir, "truthfulqa_qa.json")
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(qa_full, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Complete dataset saved to: {qa_file}")
    
    # 5. Save training set (reserved for future work)
    qa_train_file = qa_file.replace('.json', '_train.json')
    with open(qa_train_file, 'w', encoding='utf-8') as f:
        json.dump(qa_train, f, ensure_ascii=False, indent=2)
    print(f"✓ Training set saved to: {qa_train_file} (reserved for future experiments)")
    
    # 6. Save calibration set (Dcalib, for computing μ̄)
    qa_cal_file = qa_file.replace('.json', '_calibration.json')
    with open(qa_cal_file, 'w', encoding='utf-8') as f:
        json.dump(qa_calibration, f, ensure_ascii=False, indent=2)
    print(f"✓ Calibration set (Dcalib) saved to: {qa_cal_file}")
    
    print("\n" + "="*60)
    print("✓ Data preparation complete!")
    print("="*60)
    print("\nData format description:")
    print("  Each sample contains:")
    print("    - question: Original question text")
    print("    - best_answer: Best answer for training")
    print("    - mc1_choices, mc2_choices: Multiple choice options")
    print("    - correct_answers: All correct answers (for evaluation)")
    print("    - incorrect_answers: Incorrect answers (for evaluation)")
    print("\nData usage instructions:")
    print("  - *_calibration.json: For computing global μ̄ (Dcalib) - used in paper")
    print("  - *.json (without suffix): Complete dataset for final performance testing (Dtask) - used in paper")
    print("  - *_train.json: Reserved for future fine-tuning experiments (not used in current paper)")
    print("      Can be used for open-ended generation and multiple-choice tasks during evaluation")
    print("\nPrompt configuration:")
    print("  - Prompt templates (instruction, few_shot_examples) stored in data/truthfulqa/prompt.json")
    print("  - Use prompt_loader.py to load and format prompts")
    print("  - FT/Zero-shot: instruction + 6 fixed examples + question")
    print("  - ICL: instruction + 6 fixed examples + KATE examples + question")
    print("="*60)


def load_data(split: str = "all") -> Dict[str, List[Dict]]:
    """
    Load processed QA data
    
    Args:
        split: "all", "train", "calibration", or "full"
            - "all": Return all three splits
            - "calibration": Only return calibration set (Dcalib) - used for μ̄ computation
            - "full": Only return complete dataset (Dtask) - used for evaluation
            - "train": Only return training set (reserved for future work)
    
    Returns:
        If split="all": {"full": [...], "train": [...], "calibration": [...]}
        Otherwise: {"<split>": [...]}
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truthfulqa")
    base_file = os.path.join(data_dir, "truthfulqa_qa.json")
    
    result = {}
    
    if split in ["all", "full"]:
        with open(base_file, 'r', encoding='utf-8') as f:
            result["full"] = json.load(f)
    
    if split in ["all", "train"]:
        train_file = base_file.replace('.json', '_train.json')
        with open(train_file, 'r', encoding='utf-8') as f:
            result["train"] = json.load(f)
    
    if split in ["all", "calibration"]:
        cal_file = base_file.replace('.json', '_calibration.json')
        with open(cal_file, 'r', encoding='utf-8') as f:
            result["calibration"] = json.load(f)
    
    return result


if __name__ == "__main__":
    prepare_truthfulqa_qa()

