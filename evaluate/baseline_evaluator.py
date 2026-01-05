"""
Baseline Evaluator Module
Contains evaluation logic for Zero-shot, ICL, and Random-ICL
"""
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from decovec.demonstration_sampler import DemonstrationSampler


class BaselineEvaluator:
    """Baseline Evaluator"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        demonstration_sampler: DemonstrationSampler = None,
        dataset: str = "truthfulqa",
        batch_size: int = 8,
        temperature: float = 0.7,
        save_outputs: bool = False,
        output_dir: str = "results/case",
        model_name: str = None,
        self_consistency: bool = False,
        run_id: int = None
    ):
        """
        Initialize baseline evaluator
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            device: Device
            demonstration_sampler: ICL example selector (for ICL evaluation)
            dataset: Dataset type
            batch_size: Batch size
            temperature: Generation temperature (for generative tasks)
            save_outputs: Whether to save model outputs for generative datasets
            output_dir: Output file save directory
            model_name: Model name (for filename generation)
            self_consistency: Whether to enable Self-Consistency mode (uses temperature=0.7 sampling when enabled)
            run_id: Run ID (for filename in Self-Consistency mode)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.demonstration_sampler = demonstration_sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        self.model_name = model_name
        self.self_consistency = self_consistency
        self.run_id = run_id
    
    def evaluate_zero_shot(
        self,
        test_data: List[Dict],
        max_samples: int = None
    ) -> Dict[str, float]:
        """
        Evaluate Zero-shot baseline
        
        Args:
            test_data: Test data
            max_samples: Maximum number of samples
        
        Returns:
            metrics: Evaluation metrics
        """
        print("\nZero-shot Baseline Evaluation")
        print("=" * 80)
        if self.self_consistency:
            print(f"✓ Self-Consistency mode enabled (temperature=0.7, sampling)")
            if self.run_id is not None:
                print(f"  Run ID: {self.run_id}")
        
        if max_samples:
            test_data = test_data[:max_samples]
            print(f"⚡ Quick evaluation: only evaluating first {max_samples} samples")
        
        print(f"Test samples: {len(test_data)}")
        
        # Create data copy and apply Zero-shot prompt to each sample
        zero_shot_test_data = []
        print("\nPreparing Zero-shot prompts...")
        for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Preparing Zero-shot"):
            zero_shot_item = item.copy()
            zero_shot_prompt = DemonstrationSampler.construct_zero_shot_prompt(
                item,
                dataset_type=self.dataset
            )
            zero_shot_item["prompt"] = zero_shot_prompt
            zero_shot_test_data.append(zero_shot_item)
        
        print("\nStarting evaluation...")
        
        # Use dataset-specific evaluator
        evaluator = self._get_evaluator()
        metrics = self._run_evaluation(evaluator, zero_shot_test_data, eval_mode="zero_shot")
        
        print("\nZero-shot Results:")
        self._print_metrics(metrics)
        
        return metrics
    
    def evaluate_icl(
        self,
        test_data: List[Dict],
        calibration_data: List[Dict],
        precomputed_neighbors: Dict[int, List[int]] = None,
        max_samples: int = None
    ) -> Dict[str, float]:
        """
        Evaluate ICL-KATE baseline
        
        Args:
            test_data: Test data
            calibration_data: Calibration set data
            precomputed_neighbors: Precomputed neighbors
            max_samples: Maximum number of samples
        
        Returns:
            metrics: Evaluation metrics
        """
        print("\nICL Baseline Evaluation (without SVD)")
        print("=" * 80)
        if self.self_consistency:
            print(f"✓ Self-Consistency mode enabled (temperature=0.7, sampling)")
            if self.run_id is not None:
                print(f"  Run ID: {self.run_id}")
        
        if max_samples:
            test_data = test_data[:max_samples]
            print(f"⚡ Quick evaluation: only evaluating first {max_samples} samples")
        
        print(f"Test samples: {len(test_data)}")
        
        # Create data copy and apply ICL-KATE prompt
        icl_test_data = []
        print("\nPreparing ICL prompts (using precomputed neighbors)...")
        for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Preparing ICL-KATE"):
            icl_item = item.copy()
            examples = self.demonstration_sampler.get_icl_examples(
                idx, 
                calibration_data, 
                use_precomputed=True
            )
            icl_prompt = self.demonstration_sampler.construct_icl_prompt(examples, item)
            icl_item["prompt"] = icl_prompt
            icl_test_data.append(icl_item)
        
        print("\nStarting evaluation...")
        
        # Use dataset-specific evaluator
        evaluator = self._get_evaluator()
        # eval_mode includes icl method + shot to avoid output file conflicts
        eval_mode = f"icl_{self.demonstration_sampler.selection_mode}_shot{self.demonstration_sampler.n_shot}"
        metrics = self._run_evaluation(evaluator, icl_test_data, eval_mode=eval_mode)
        
        print(f"\nICL ({self.demonstration_sampler.selection_mode}, shot={self.demonstration_sampler.n_shot}) Results:")
        self._print_metrics(metrics)
        
        return metrics
    
    def evaluate_random_icl(
        self,
        test_data: List[Dict],
        calibration_data: List[Dict],
        max_samples: int = None
    ) -> Dict[str, float]:
        """
        Evaluate Random ICL baseline
        
        Args:
            test_data: Test data
            calibration_data: Calibration set data
            max_samples: Maximum number of samples
        
        Returns:
            metrics: Evaluation metrics
        """
        print("\nRandom ICL Baseline Evaluation (without SVD)")
        print("=" * 80)
        if self.self_consistency:
            print(f"✓ Self-Consistency mode enabled (temperature=0.7, sampling)")
            if self.run_id is not None:
                print(f"  Run ID: {self.run_id}")
        
        if max_samples:
            test_data = test_data[:max_samples]
            print(f"⚡ Quick evaluation: only evaluating first {max_samples} samples")
        
        print(f"Test samples: {len(test_data)}")
        
        # Create data copy and apply random ICL prompt
        random_icl_test_data = []
        print("\nPreparing Random ICL prompts...")
        for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Preparing Random ICL"):
            random_icl_item = item.copy()
            examples = self._get_random_icl_examples(idx, calibration_data)
            random_icl_prompt = self.demonstration_sampler.construct_icl_prompt(examples, item)
            random_icl_item["prompt"] = random_icl_prompt
            random_icl_test_data.append(random_icl_item)
        
        print("\nStarting evaluation...")
        
        # Use dataset-specific evaluator
        evaluator = self._get_evaluator()
        # eval_mode includes shot to avoid output file conflicts for different shots
        eval_mode = f"random_icl_shot{self.demonstration_sampler.n_shot}"
        metrics = self._run_evaluation(evaluator, random_icl_test_data, eval_mode=eval_mode)
        
        print("\nRandom ICL Results:")
        self._print_metrics(metrics)
        
        return metrics
    
    def _get_random_icl_examples(
        self,
        query_idx: int,
        calibration_data: List[Dict],
        seed: int = None
    ) -> List[Dict]:
        """
        Randomly select ICL examples
        
        Args:
            query_idx: Query sample index
            calibration_data: Calibration set data
            seed: Random seed
        
        Returns:
            examples: Randomly selected example list
        """
        if seed is None:
            seed = query_idx
        
        rng = np.random.RandomState(seed)
        available_indices = list(range(len(calibration_data)))
        n_examples = min(self.demonstration_sampler.n_shot, len(available_indices))
        selected_indices = rng.choice(available_indices, size=n_examples, replace=False)
        
        examples = [calibration_data[i] for i in selected_indices]
        return examples
    
    def _get_evaluator(self):
        """Get dataset-specific evaluator"""
        if self.dataset == "math500" or self.dataset == "math500_pool_ablation":
            from evaluate.evaluate_math500 import Math500Evaluator
            return Math500Evaluator(self.tokenizer, self.device, model_name=self.model_name)
        elif self.dataset == "aqua_rat" or self.dataset == "aqua_rat_pool_ablation":
            from evaluate.evaluate_aqua_rat import AquaRATEvaluator
            return AquaRATEvaluator(self.tokenizer, self.device, model_name=self.model_name)
        else:
            # truthfulqa and truthfulqa_pool_ablation
            from evaluate.evaluate_truthfulqa import QAEvaluator
            return QAEvaluator(self.tokenizer, self.device, dataset_name=self.dataset)
    
    def _run_evaluation(self, evaluator, test_data: List[Dict], eval_mode: str = "unknown") -> Dict[str, float]:
        """Run evaluation"""
        if self.dataset in {"truthfulqa", "truthfulqa_pool_ablation"}:
            compute_mc2 = True
            compute_mc3 = True
            return evaluator.evaluate_dataset(
                self.model,
                test_data,
                compute_mc1=True,
                compute_mc2=compute_mc2,
                compute_mc3=compute_mc3,
                batch_size=self.batch_size,
                max_samples=None,
                use_batch_compute=True
            )
        else:
            # For generative tasks, pass temperature and save options
            # Check if evaluator supports temperature and save_outputs parameters
            import inspect
            sig = inspect.signature(evaluator.evaluate_dataset)
            kwargs = {}
            
            # Determine sampling parameters based on self_consistency mode
            if self.self_consistency:
                kwargs['temperature'] = 0.7
                kwargs['do_sample'] = True
            else:
                # Default greedy decoding
                if 'temperature' in sig.parameters:
                    kwargs['temperature'] = self.temperature
                kwargs['do_sample'] = False
            
            if 'save_outputs' in sig.parameters:
                kwargs['save_outputs'] = self.save_outputs
            if 'output_dir' in sig.parameters:
                kwargs['output_dir'] = self.output_dir
            if 'eval_mode' in sig.parameters:
                kwargs['eval_mode'] = eval_mode
            if 'self_consistency' in sig.parameters:
                kwargs['self_consistency'] = self.self_consistency
            if 'run_id' in sig.parameters:
                kwargs['run_id'] = self.run_id
            
            if kwargs:
                return evaluator.evaluate_dataset(
                    self.model,
                    test_data,
                    max_samples=None,
                    **kwargs
                )
            else:
                return evaluator.evaluate_dataset(
                    self.model,
                    test_data,
                    max_samples=None
                )
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """Print evaluation metrics"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")

