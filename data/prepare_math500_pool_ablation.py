"""
Math-500 pool size ablation data preparation script

- Test set: Reuse processing logic from math-500/test.jsonl
- Example pool: Stratified sampling from MATH/test (200 per level, 1000 total)
"""
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple
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
        print(f"✓ Data loaded successfully: {len(data)} examples")
    except Exception as exc:
        print(f"✗ Data loading failed: {exc}")
        return []
    return data


def format_math500_example(example: Dict) -> Optional[Dict]:
    """Format Math-500 sample, keeping question, chain-of-thought, and final answer."""
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
    """Randomly sample n examples from data."""
    random.seed(seed)
    if n_samples >= len(data):
        print(f"⚠️  Requested samples ({n_samples}) >= dataset size ({len(data)}), returning all data")
        return data.copy()
    sampled = random.sample(data, n_samples)
    print(f"✓ Randomly sampled {len(sampled)} examples (seed={seed})")
    return sampled


def extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract the last \boxed answer from solution."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", solution)
    if boxed:
        return boxed[-1].strip()
    return None


def load_math_pool_examples(pool_root: str) -> List[Tuple[str, Dict]]:
    if not os.path.isdir(pool_root):
        raise FileNotFoundError(f"MATH test set directory not found: {pool_root}")
    items: List[Tuple[str, Dict]] = []
    for root, _, files in os.walk(pool_root):
        for name in files:
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items.append((path, data))
    if not items:
        raise ValueError("No JSON samples found in MATH test set")
    return items


def format_math_pool_example(example: Dict, uid: str) -> Dict:
    problem = example.get("problem", "").strip()
    solution = example.get("solution", "").strip()
    level = example.get("level", "").strip()
    subject = example.get("type", "").strip()

    if not problem or not solution or not level:
        raise ValueError(f"Sample missing required fields: {uid}")

    final_answer = extract_boxed_answer(solution)
    if final_answer is None or final_answer == "":
        raise ValueError(f"Unable to extract answer from solution: {uid}")

    return {
        "question": problem,
        "solution": solution,
        "final_answer": final_answer,
        "subject": subject,
        "level": level,
        "unique_id": uid,
    }


def stratified_sample_pool(pool_items: List[Tuple[str, Dict]], target_per_level: int, seed: int, base_dir: str) -> List[Dict]:
    rng = random.Random(seed)
    by_level: Dict[str, List[Tuple[str, Dict]]] = {}
    for path, data in pool_items:
        level = data.get("level", "").strip()
        by_level.setdefault(level, []).append((path, data))

    for level, samples in by_level.items():
        if len(samples) < target_per_level:
            raise ValueError(f"Level {level} available samples {len(samples)} < target {target_per_level}")

    selected: List[Dict] = []
    for level, samples in sorted(by_level.items()):
        rng.shuffle(samples)
        chosen = samples[:target_per_level]
        for path, data in chosen:
            uid = os.path.relpath(path, base_dir)
            selected.append(format_math_pool_example(data, uid))
    return selected


def prepare_math500(calibration_size: int = 100, test_size: int = 256, pool_target_per_level: int = 200) -> None:
    """Main function: Generate Math-500 test set + MATH stratified example pool."""
    config = get_config()

    print("\n" + "=" * 60)
    print("Processing Math-500 pool size ablation dataset")
    print("=" * 60)

    # Test set (consistent with original logic)
    raw_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "original_dataset",
        "math-500",
        "test.jsonl",
    )
    raw_data = load_math500_jsonl(raw_file)
    if not raw_data:
        print("✗ Failed to load math-500 test data, exiting")
        return

    if config.data.use_full_test_set:
        test_raw = raw_data
        print("✓ Using full Math-500 test set")
    else:
        test_raw = sample_data(raw_data, test_size, seed=config.seed + 1)

    test_data: List[Dict] = []
    for example in tqdm(test_raw, desc="Formatting test set"):
        formatted = format_math500_example(example)
        if formatted:
            test_data.append(formatted)

    # Example pool (stratified sampling, 200 per level, 1000 total)
    pool_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "original_dataset",
        "MATH",
        "test",
    )
    pool_items = load_math_pool_examples(pool_root)
    pool_data = stratified_sample_pool(pool_items, pool_target_per_level, config.seed, pool_root)

    if len(pool_data) != pool_target_per_level * 5:
        raise ValueError(f"示例池期望 {pool_target_per_level*5}，当前 {len(pool_data)}")

    print(f"\n✓ 示例池分层完成，每个 level {pool_target_per_level} 条")
    level_counts = {}
    for item in pool_data:
        level_counts[item["level"]] = level_counts.get(item["level"], 0) + 1
    print("  Level 分布: " + ", ".join(f"{lvl}:{cnt}" for lvl, cnt in sorted(level_counts.items())))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math500_pool_ablation")
    os.makedirs(output_dir, exist_ok=True)

    test_out = os.path.join(output_dir, "math500_pool_ablation.json")
    cal_out = os.path.join(output_dir, "math500_pool_ablation_calibration.json")

    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    with open(cal_out, "w", encoding="utf-8") as f:
        json.dump(pool_data, f, ensure_ascii=False, indent=2)

    print("\nData statistics:")
    print(f"  - Test set: {len(test_data)}")
    print(f"  - Example pool: {len(pool_data)}")
    print(f"  - Example pool avg solution length: {sum(len(x['solution']) for x in pool_data)/len(pool_data):.1f}")

    print("\nSample examples (first 2 from test set):")
    for idx, item in enumerate(test_data[:2], start=1):
        print(f"  Sample {idx}: {item['question'][:120]}{'...' if len(item['question'])>120 else ''}")
        print(f"    Answer: {item['final_answer']}")
        print(f"    Subject/Level: {item.get('subject','')} / {item.get('level','')}")

    print("\n" + "=" * 60)
    print("✓ Math-500 pool size ablation data generation complete")
    print(f"  Test set -> {test_out}")
    print(f"  Example pool -> {cal_out}")
    print("=" * 60)


def load_data(split: str = "all") -> Dict[str, List[Dict]]:
    """Load Math-500 pool size ablation data."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math500_pool_ablation")
    result: Dict[str, List[Dict]] = {}

    if split in {"all", "test"}:
        with open(os.path.join(data_dir, "math500_pool_ablation.json"), "r", encoding="utf-8") as f:
            result["test"] = json.load(f)
    if split in {"all", "calibration"}:
        with open(os.path.join(data_dir, "math500_pool_ablation_calibration.json"), "r", encoding="utf-8") as f:
            result["calibration"] = json.load(f)
    return result


if __name__ == "__main__":
    prepare_math500()

