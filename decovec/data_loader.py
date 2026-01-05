"""
Data Loading Module

Unified management of loading and splitting different datasets.
"""
import os
import json
import random
from typing import Dict, List, Tuple


class DataLoader:
    """Data Loader"""
    
    def __init__(self, dataset: str = "truthfulqa"):
        """
        Initialize the data loader
        
        Args:
            dataset: Dataset type
        """
        self.dataset = dataset
        # Limit the actual calibration set size for example selection; None means use all
        self.calibration_pool_size = None
        # Random seed for shuffling before truncation; None means no shuffling
        self.calibration_pool_seed = None
        self.data_dir = self._get_data_dir()
    
    def _get_data_dir(self) -> str:
        """Get dataset directory"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "data", self.dataset)
    
    def load_splits(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load calibration and test sets
        
        Returns:
            (calibration_data, test_data)
        """
        print("\nLoading data...")
        
        if self.dataset == "math500":
            calibration_data, test_data = self._load_math500()
        elif self.dataset == "math500_pool_ablation":
            calibration_data, test_data = self._load_math500_pool_ablation()
        elif self.dataset == "aqua_rat":
            calibration_data, test_data = self._load_aqua_rat()
        elif self.dataset == "aqua_rat_pool_ablation":
            calibration_data, test_data = self._load_aqua_rat_pool_ablation()
        else:
            calibration_data, test_data = (
                self._load_truthfulqa()
                if self.dataset == "truthfulqa"
                else self._load_truthfulqa_pool_ablation()
            )

        calibration_data = self._apply_calibration_cap(calibration_data)
        return calibration_data, test_data

    def _apply_calibration_cap(self, calibration_data: List[Dict]) -> List[Dict]:
        """Truncate calibration set based on calibration_pool_size."""
        if self.calibration_pool_size is None:
            return calibration_data
        if self.calibration_pool_size <= 0:
            return calibration_data

        data = calibration_data
        if self.calibration_pool_seed is not None and len(calibration_data) > self.calibration_pool_size:
            rng = random.Random(self.calibration_pool_seed)
            data = calibration_data.copy()
            rng.shuffle(data)

        capped = data[: self.calibration_pool_size]
        if len(capped) < len(calibration_data):
            print(
                f"✓ Using calibration examples: {len(capped)}/{len(calibration_data)}"
                + (f" (seed={self.calibration_pool_seed})" if self.calibration_pool_seed is not None else "")
            )
        return capped
    
    def _load_truthfulqa(self) -> Tuple[List[Dict], List[Dict]]:
        """Load TruthfulQA dataset"""
        # Load full dataset
        full_file = os.path.join(self.data_dir, "truthfulqa_qa.json")
        with open(full_file, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        # Load calibration set
        cal_file = os.path.join(self.data_dir, "truthfulqa_qa_calibration.json")
        with open(cal_file, 'r', encoding='utf-8') as f:
            calibration_data = json.load(f)
        
        # Create question set from calibration for exclusion
        calibration_questions = {item["question"] for item in calibration_data}
        
        # Test set = Full dataset - Calibration set
        test_data = [item for item in full_data if item["question"] not in calibration_questions]
        
        print(f"✓ Dataset: TruthfulQA")
        print(f"✓ Full dataset: {len(full_data)} samples")
        print(f"✓ Calibration set (Dcalib): {len(calibration_data)} samples ({len(calibration_data)/len(full_data)*100:.1f}%)")
        print(f"✓ Test set (Dtest): {len(test_data)} samples ({len(test_data)/len(full_data)*100:.1f}%)")
        
        # Validation
        assert len(test_data) + len(calibration_data) == len(full_data), \
            f"Data split error: {len(test_data)} + {len(calibration_data)} != {len(full_data)}"
        
        return calibration_data, test_data

    def _load_truthfulqa_pool_ablation(self) -> Tuple[List[Dict], List[Dict]]:
        """Load TruthfulQA pool size ablation dataset."""
        test_file = os.path.join(self.data_dir, "truthfulqa_pool_ablation.json")
        cal_file = os.path.join(self.data_dir, "truthfulqa_pool_ablation_calibration.json")

        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        with open(cal_file, 'r', encoding='utf-8') as f:
            calibration_data = json.load(f)

        print(f"✓ Dataset: TruthfulQA Pool Ablation")
        print(f"✓ Calibration set (example pool): {len(calibration_data)} samples")
        print(f"✓ Test set (Dtest): {len(test_data)} samples")

        return calibration_data, test_data

    def _load_math500(self) -> Tuple[List[Dict], List[Dict]]:
        """Load Math-500 dataset"""
        return self._load_generic_split(
            dataset_name="Math-500",
            test_filename="math500.json",
            cal_filename="math500_calibration.json"
        )

    def _load_math500_pool_ablation(self) -> Tuple[List[Dict], List[Dict]]:
        """Load Math-500 pool size ablation dataset"""
        return self._load_generic_split(
            dataset_name="Math-500 Pool Ablation",
            test_filename="math500_pool_ablation.json",
            cal_filename="math500_pool_ablation_calibration.json"
        )

    def _load_aqua_rat(self) -> Tuple[List[Dict], List[Dict]]:
        """Load AQUA-RAT dataset"""
        return self._load_generic_split(
            dataset_name="AQUA-RAT",
            test_filename="aqua_rat.json",
            cal_filename="aqua_rat_calibration.json"
        )

    def _load_aqua_rat_pool_ablation(self) -> Tuple[List[Dict], List[Dict]]:
        """Load AQUA-RAT pool size ablation dataset"""
        return self._load_generic_split(
            dataset_name="AQUA-RAT Pool Ablation",
            test_filename="aqua_rat_pool_ablation.json",
            cal_filename="aqua_rat_pool_ablation_calibration.json"
        )

    def _load_generic_split(
        self,
        dataset_name: str,
        test_filename: str,
        cal_filename: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generic JSON split loader"""
        test_file = os.path.join(self.data_dir, test_filename)
        cal_file = os.path.join(self.data_dir, cal_filename)

        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        with open(cal_file, 'r', encoding='utf-8') as f:
            calibration_data = json.load(f)

        print(f"✓ Dataset: {dataset_name}")
        print(f"✓ Calibration set (Dcalib): {len(calibration_data)} samples")
        print(f"✓ Test set (Dtest): {len(test_data)} samples")

        return calibration_data, test_data

