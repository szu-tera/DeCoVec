"""
Math-500 dataset loading and preprocessing
For DeCoVec experiments

Data description:
- Data source: /data/original_dataset/math-500/test.jsonl
- Each sample contains problem / solution / answer / subject / level
- Goal: Generate ICL data with chain-of-thought (solution) and final answer
"""
import json
import os
import random
from typing import Dict, List, Optional
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config


def load_math500_jsonl(file_path: str) -> List[Dict]:
    """Load Math-500 data from JSONL file."""
    print(f"Loading Math-500 data: {file_path}")
    data: List[Dict] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        print(f"✓ Data loaded successfully: {len(data)} samples")
    except Exception as exc:
        print(f"✗ Data loading failed: {exc}")
        return []
    return data


def format_math500_example(example: Dict) -> Optional[Dict]:
    """Format Math-500 sample, preserving problem, solution, and final answer."""
    problem = example.get("problem", "").strip()
    solution = example.get("solution", "").strip()
    answer = example.get("answer", "").strip()

    if not problem or not solution or not answer:
        return None

    return {
        "question": problem,
        "solution": solution,
        "final_answer": answer,
        "subject": example.get("subject", ""),
        "level": example.get("level", ""),
        "unique_id": example.get("unique_id", ""),
    }


def sample_data(data: List[Dict], n_samples: int, seed: int) -> List[Dict]:
    """Randomly sample n samples from data."""
    random.seed(seed)
    if n_samples >= len(data):
        print(f"⚠️  Requested samples ({n_samples}) >= dataset size ({len(data)}), returning all data")
        return data.copy()
    sampled = random.sample(data, n_samples)
    print(f"✓ Randomly sampled {len(sampled)} samples (seed={seed})")
    return sampled


def prepare_math500(calibration_size: int = 100, test_size: int = 256) -> None:
    """Main function: Load and preprocess Math-500 data."""
    config = get_config()

    print("\n" + "=" * 60)
    print("Processing Math-500 dataset")
    print("=" * 60)

    raw_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "original_dataset",
        "math-500",
        "test.jsonl",
    )

    raw_data = load_math500_jsonl(raw_file)
    if not raw_data:
        print("✗ Failed to load raw data, exiting")
        return

    calibration_raw = sample_data(raw_data, calibration_size, seed=config.seed)
    if config.data.use_full_test_set:
        print("✓ Using complete Math-500 set as test data (no sampling)")
        test_raw = raw_data
    else:
        test_raw = sample_data(raw_data, test_size, seed=config.seed + 1)

    print("\nFormatting data...")
    calibration_data: List[Dict] = []
    for example in tqdm(calibration_raw, desc="Formatting calibration set"):
        formatted = format_math500_example(example)
        if formatted:
            calibration_data.append(formatted)

    test_data: List[Dict] = []
    for example in tqdm(test_raw, desc="Formatting test set"):
        formatted = format_math500_example(example)
        if formatted:
            test_data.append(formatted)

    print(f"✓ Calibration set: {len(calibration_data)} samples")
    print(f"✓ Test set: {len(test_data)} samples")

    print("\nData statistics:")
    def report_stats(name: str, dataset: List[Dict]) -> None:
        subjects = {}
        for item in dataset:
            subject = item.get("subject", "Unknown")
            subjects[subject] = subjects.get(subject, 0) + 1
        avg_solution_len = (
            sum(len(item["solution"]) for item in dataset) / len(dataset)
            if dataset else 0
        )
        print(f"  - {name} samples: {len(dataset)}")
        print(f"    · Average solution length: {avg_solution_len:.1f} chars")
        if subjects:
            top_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)[:3]
            subject_str = ", ".join(f"{sub}:{cnt}" for sub, cnt in top_subjects)
            print(f"    · Top 3 subjects: {subject_str}")

    report_stats("Calibration", calibration_data)
    report_stats("Test", test_data)

    print("\nSample examples (first 2 from test set):")
    for idx, item in enumerate(test_data[:2], start=1):
        print(f"\n  Sample {idx}:")
        print(f"  Problem: {item['question'][:120]}{'...' if len(item['question']) > 120 else ''}")
        print(f"  Final answer: {item['final_answer']}")
        print(f"  Subject/Level: {item.get('subject','')} / {item.get('level','')}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math500")
    os.makedirs(output_dir, exist_ok=True)

    test_out = os.path.join(output_dir, "math500.json")
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Test set saved to: {test_out}")

    cal_out = os.path.join(output_dir, "math500_calibration.json")
    with open(cal_out, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Calibration set saved to: {cal_out}")

    print("\n" + "=" * 60)
    print("✓ Math-500 data preparation complete!")
    print("=" * 60)


def load_data(split: str = "all") -> Dict[str, List[Dict]]:
    """Load processed Math-500 data."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math500")
    result: Dict[str, List[Dict]] = {}

    if split in {"all", "test"}:
        with open(os.path.join(data_dir, "math500.json"), "r", encoding="utf-8") as f:
            result["test"] = json.load(f)
    if split in {"all", "calibration"}:
        with open(os.path.join(data_dir, "math500_calibration.json"), "r", encoding="utf-8") as f:
            result["calibration"] = json.load(f)
    return result


if __name__ == "__main__":
    prepare_math500()

