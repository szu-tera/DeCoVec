"""
AQUA-RAT dataset preprocessing script

- Uses tokenized split (validation -> calibration, test -> test)
- Generates unified generative reasoning format (with Options / rationale / final answer)
- Output:
    data/aqua_rat/aqua_rat.json
    data/aqua_rat/aqua_rat_calibration.json
"""
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config  # type: ignore


CHOICE_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return text.strip()


def parse_option(option: str, fallback_label: str) -> Tuple[str, str]:
    """
    Parse format like 'A ) 5' / 'A)5' into (label, text)
    """
    option = option.strip()
    if ")" in option:
        label_part, text_part = option.split(")", 1)
        label = label_part.strip().strip("(").strip().rstrip(".").upper()
        text = text_part.strip()
        if not label:
            label = fallback_label
    else:
        label = fallback_label
        text = option
    return label, text


def load_split(parquet_path: str) -> List[Dict]:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    return df.to_dict("records")


def sample_subset(data: List[Dict], target_size: int, seed: int) -> List[Dict]:
    if len(data) <= target_size:
        return data
    rng = random.Random(seed)
    return rng.sample(data, target_size)


def format_aqua_item(example: Dict, sample_idx: int) -> Dict:
    question = normalize_text(example.get("question", ""))
    rationale = normalize_text(example.get("rationale", ""))
    raw_options = example.get("options", [])
    if question == "" or len(raw_options) == 0:
        return {}
    
    options: List[str] = []
    labels: List[str] = []
    answer_mapping: Dict[str, str] = {}
    
    for idx, raw in enumerate(raw_options):
        fallback = CHOICE_LABELS[idx]
        label, text = parse_option(str(raw), fallback)
        labels.append(label)
        options.append(text)
        answer_mapping[label.upper()] = text
    
    answer_label = str(example.get("correct", "")).strip().upper()
    if answer_label not in answer_mapping:
        # Try using the first option as fallback
        answer_label = labels[0].upper()
    
    answer_text = answer_mapping.get(answer_label, "")
    final_answer = answer_label
    
    formatted_answer = rationale if rationale else f"The correct answer is {answer_label}."
    if "####" not in formatted_answer:
        formatted_answer = formatted_answer.rstrip() + f"\n#### {final_answer}"
    
    return {
        "id": sample_idx,
        "question": question,
        "choices": options,
        "labels": labels,
        "answer": formatted_answer,
        "final_answer": final_answer,
        "answer_label": final_answer,
        "answer_text": answer_text,
        "rationale": rationale,
    }


def save_split(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ File saved: {path}")


def prepare_aqua_rat():
    config = get_config()
    seed = config.seed
    
    raw_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "original_dataset",
        "aqua_rat",
        "tokenized"
    )
    
    val_file = os.path.join(raw_dir, "validation-00000-of-00001.parquet")
    test_file = os.path.join(raw_dir, "test-00000-of-00001.parquet")
    
    print("Loading validation split (as calibration set)")
    val_examples = load_split(val_file)
    print("Loading test split (as test set)")
    test_examples = load_split(test_file)
    
    # Display raw input example
    print("\n" + "=" * 80)
    print("Raw input example (validation split):")
    print("=" * 80)
    if val_examples:
        raw_example = val_examples[0]
        print(f"Raw data fields: {list(raw_example.keys())}")
        print(f"\nSample raw data:")
        print(f"  question: {raw_example.get('question', '')[:200]}...")
        print(f"  options: {raw_example.get('options', [])}")
        print(f"  correct: {raw_example.get('correct', '')}")
        print(f"  rationale: {raw_example.get('rationale', '')[:200] if raw_example.get('rationale') else 'N/A'}...")
    
    val_examples = sample_subset(val_examples, 100, seed)
    test_examples = sample_subset(test_examples, 256, seed + 1)
    
    calibration_data: List[Dict] = []
    first_formatted = None
    for idx, item in enumerate(tqdm(val_examples, desc="Processing calibration set")):
        formatted = format_aqua_item(item, idx)
        if formatted:
            calibration_data.append(formatted)
            if first_formatted is None:
                first_formatted = (item, formatted)
    
    test_data: List[Dict] = []
    for idx, item in enumerate(tqdm(test_examples, desc="Processing test set")):
        formatted = format_aqua_item(item, idx)
        if formatted:
            test_data.append(formatted)
    
    # Display processed example
    if first_formatted:
        raw_item, formatted_item = first_formatted
        print("\n" + "=" * 80)
        print("Processed example (input -> output):")
        print("=" * 80)
        print("\n[Raw Input]")
        print(f"  question: {raw_item.get('question', '')[:150]}...")
        print(f"  options: {raw_item.get('options', [])}")
        print(f"  correct: {raw_item.get('correct', '')}")
        print(f"  rationale: {raw_item.get('rationale', '')[:150] if raw_item.get('rationale') else 'N/A'}...")
        print("\n[Processed Output]")
        print(f"  id: {formatted_item.get('id')}")
        print(f"  question: {formatted_item.get('question', '')[:150]}...")
        print(f"  choices: {formatted_item.get('choices', [])}")
        print(f"  labels: {formatted_item.get('labels', [])}")
        print(f"  answer_label: {formatted_item.get('answer_label')}")
        print(f"  final_answer: {formatted_item.get('final_answer')}")
        print(f"  answer_text: {formatted_item.get('answer_text', '')[:100]}...")
        print(f"  answer (full format): {formatted_item.get('answer', '')[:200]}...")
        print(f"  rationale: {formatted_item.get('rationale', '')[:150] if formatted_item.get('rationale') else 'N/A'}...")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aqua_rat")
    os.makedirs(output_dir, exist_ok=True)
    
    cal_path = os.path.join(output_dir, "aqua_rat_calibration.json")
    test_path = os.path.join(output_dir, "aqua_rat.json")
    
    save_split(calibration_data, cal_path)
    save_split(test_data, test_path)
    
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    print(f"  - Calibration samples: {len(calibration_data)}")
    print(f"  - Test samples: {len(test_data)}")
    if test_data:
        example = test_data[0]
        print(f"\nTest set example:")
        print(f"  Question: {example['question'][:120]}...")
        print(f"  Number of choices: {len(example['choices'])}")
        print(f"  Choice labels: {example['labels']}")
        print(f"  Correct answer: {example['final_answer']}")
        print(f"  Answer text: {example['answer_text'][:100] if example.get('answer_text') else 'N/A'}...")
    
    print("\n✓ AQUA-RAT data preparation complete!")


if __name__ == "__main__":
    prepare_aqua_rat()

