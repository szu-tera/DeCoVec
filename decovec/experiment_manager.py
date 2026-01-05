"""
Experiment Manager Module

Coordinates various modules and manages experiment workflow.
"""
import os
import json
import torch
from datetime import datetime
from typing import Dict, List, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from simple_config import get_config
from decovec.data_loader import DataLoader
from decovec.demonstration_sampler import DemonstrationSampler
from decovec.task_vector_builder import TaskVectorComputer
from decovec.scale_tester import ScaleTester
from evaluate.evaluator_factory import EvaluatorFactory


class ExperimentManager:
    """Experiment Manager"""
    
    # Generative datasets list (only datasets used in paper)
    GENERATIVE_DATASETS = {"math500", "aqua_rat", "math500_pool_ablation", "aqua_rat_pool_ablation"}
    
    def __init__(
        self,
        dataset: str = "truthfulqa",
        n_shot: int = None,
        icl_method: str = "kate",
        save_outputs: bool = False,
        output_dir: str = "results/case",
        self_consistency: bool = False,
        run_id: int = None,
        example_order: str = "ordered",
        example_order_seed: int = None
    ):
        """
        Initialize the experiment manager
        
        Args:
            dataset: Dataset type
            n_shot: Number of ICL examples (if None, automatically set based on dataset)
            save_outputs: Whether to save model outputs for generative datasets
            output_dir: Output file save directory
            self_consistency: Whether to enable Self-Consistency mode (uses temperature=0.7 sampling)
            run_id: Run iteration ID (for file naming in Self-Consistency mode)
        """
        self.config = get_config()
        self.dataset = dataset
        self.device = self.config.device
        
        # Set n_shot (based on dataset type)
        if n_shot is not None:
            self.n_shot = n_shot
        elif dataset in self.GENERATIVE_DATASETS:
            self.n_shot = 10  # Generative tasks use 10 shots
        else:
            self.n_shot = 15  # truthfulqa etc. use 15 shots
        
        self.max_demo_tokens = 2048
        
        print(f"✓ Dataset '{dataset}' ICL config: n_shot={self.n_shot}, max_demo_tokens={self.max_demo_tokens}")
        
        # Create data loader
        self.data_loader = DataLoader(dataset)
        
        # Model related
        self.model = None
        self.tokenizer = None
        self.emb_model = None
        self.emb_model_identifier = None
        
        # Core components
        self.demonstration_sampler = None
        self.task_vector_builder = None
        self.baseline_evaluator = None
        # decovec_evaluator deprecated, use scale_tester instead
        self.scale_tester = None
        self.icl_method = icl_method
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        self.self_consistency = self_consistency
        self.run_id = run_id
        self.example_order = example_order
        self.example_order_seed = example_order_seed
    
    def setup_models(self):
        """Load all required models"""
        print("\n" + "=" * 80)
        print("Loading Models")
        print("=" * 80)
        
        # 1. Load tokenizer
        print("Loading tokenizer...")
        # Resolve relative paths to absolute paths under project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = self.config.model_path or self.config.model_name
        if model_path and not os.path.isabs(model_path) and not model_path.startswith('http'):
            model_path = os.path.join(project_root, model_path)
            print(f"  Using local model: {model_path}")
        
        is_local = model_path and os.path.exists(model_path) and os.path.isdir(model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=is_local
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer loaded")
        
        # 2. Load language model
        print("\nLoading language model...")
        torch_dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            local_files_only=is_local
        )
        self.model.eval()
        print(f"✓ Language model loaded")
        
        # 3. Load embedding model
        print("\nLoading Embedding model...")
        emb_model_path = self.config.emb_model_path
        # Resolve relative paths to absolute paths under project root
        if emb_model_path and not os.path.isabs(emb_model_path):
            emb_model_path = os.path.join(project_root, emb_model_path)
            self.config.emb_model_path = emb_model_path
        
        if not os.path.exists(emb_model_path):
            print(f"⚠️  Local model not found: {emb_model_path}")
            print(f"Using online model: {self.config.emb_model_name}")
            self.emb_model = SentenceTransformer(self.config.emb_model_name)
            self.emb_model_identifier = self.config.emb_model_name
        else:
            try:
                self.emb_model = SentenceTransformer(emb_model_path)
                print(f"✓ Loaded from local: {emb_model_path}")
                self.emb_model_identifier = emb_model_path
            except Exception as e:
                print(f"⚠️  Local loading failed: {e}")
                print(f"Using online model: {self.config.emb_model_name}")
                self.emb_model = SentenceTransformer(self.config.emb_model_name)
                self.emb_model_identifier = self.config.emb_model_name
        
        print("✓ Embedding model loaded")
        
        # 4. Create core components
        print("\nCreating core components...")
        
        # ICL demonstration sampler
        self.demonstration_sampler = DemonstrationSampler(
            emb_model=self.emb_model,
            n_shot=self.n_shot,
            max_demo_tokens=self.max_demo_tokens,
            tokenizer=self.tokenizer,
            dataset_type=self.dataset,
            selection_mode=self.icl_method,
            example_order=self.example_order,
            example_order_seed=self.example_order_seed
        )
        print("✓ ICL demonstration sampler created")
        
        # DeCoVec calculator
        self.task_vector_builder = TaskVectorComputer(
            model=self.model,
            tokenizer=self.tokenizer,
            demonstration_sampler=self.demonstration_sampler,
            device=self.device,
            dataset=self.dataset
        )
        print("✓ DeCoVec calculator created")
        
        # Get model name
        model_name = self.config.model_name or "unknown_model"
        
        # Create evaluator
        self.baseline_evaluator = EvaluatorFactory.create_baseline_evaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            demonstration_sampler=self.demonstration_sampler,
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            temperature=self.config.temperature,
            save_outputs=self.save_outputs,
            output_dir=self.output_dir,
            model_name=model_name,
            self_consistency=self.self_consistency,
            run_id=self.run_id
        )
        print("✓ Baseline evaluator created")
        
        # Create steering computer (for dynamic DeCoVec computation)
        from decovec.decovec_core import TaskVectorBuilder
        steering_computer = TaskVectorBuilder()
        
        # λ value tester
        # Get model name
        model_name = self.config.model_name or "unknown_model"
        
        self.scale_tester = ScaleTester(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            demonstration_sampler=self.demonstration_sampler,
            task_vector_builder=self.task_vector_builder,
            dataset=self.dataset,
            save_outputs=self.save_outputs,
            output_dir=self.output_dir,
            model_name=model_name,
            self_consistency=self.self_consistency,
            run_id=self.run_id,
            icl_method=self.icl_method
        )
        print("✓ λ value tester created")
        
        print("=" * 80)
    
    def build_knn_index(self, calibration_data: List[Dict]) -> Tuple[Any, Any]:
        """Build index (kNN or BM25, depending on ICL method)"""
        # Choose different index types based on ICL method
        if self.icl_method == "bm25":
            # Build BM25 index
            bm25_index = self.demonstration_sampler.build_bm25_index(
                calibration_data,
                use_cache=False
            )
            
            # Return None, bm25_index to maintain interface consistency
            return None, bm25_index
        else:
            # Build kNN index (original logic)
            embeddings, knn_index = self.demonstration_sampler.build_knn_index(
                calibration_data,
                use_cache=False
            )
            
            return embeddings, knn_index
    
    def precompute_test_neighbors(
        self,
        test_data: List[Dict],
        calibration_data: List[Dict]
    ) -> Dict[int, List[int]]:
        """Precompute nearest neighbors for test set"""
        precomputed_neighbors = self.demonstration_sampler.precompute_all_neighbors(
            test_data,
            calibration_data,
            max_neighbors=self.n_shot * 2
        )
        
        return precomputed_neighbors

    def set_icl_method(self, icl_method: str):
        """Set the current ICL example selection method"""
        self.icl_method = icl_method
        if self.demonstration_sampler is not None:
            self.demonstration_sampler.set_selection_mode(icl_method)
        if self.baseline_evaluator is not None:
            self.baseline_evaluator.demonstration_sampler = self.demonstration_sampler
        if self.task_vector_builder is not None:
            self.task_vector_builder.demonstration_sampler = self.demonstration_sampler
        # decovec_evaluator deprecated, removed
        if self.scale_tester is not None:
            self.scale_tester.demonstration_sampler = self.demonstration_sampler
            self.scale_tester.icl_method = icl_method
    
    def compute_delta_z_with_cache(
        self,
        calibration_data: List[Dict],
        knn_index: Any,
        icl_method: str = "kate"
    ) -> Dict:
        """
        Compute delta-z
        
        Args:
            calibration_data: Calibration set data
            knn_index: kNN index (Note: may need to rebuild if icl_method differs from current selector mode)
            icl_method: ICL method for constructing task vector (may differ from current selector mode)
        """
        # Save current selection_mode
        original_selection_mode = self.demonstration_sampler.selection_mode
        
        # If icl_method differs from current selector mode, temporarily switch
        # Note: Different ICL methods (e.g., bm25 vs kate) may need different indices
        # Here we assume index is already built, just switch selection mode
        if icl_method != original_selection_mode:
            print(f"  Temporarily switching ICL selection mode: {original_selection_mode} -> {icl_method} (for task vector construction)")
            self.demonstration_sampler.set_selection_mode(icl_method)
            # If switching to bm25, ensure bm25_index is built
            if icl_method == "bm25" and self.demonstration_sampler.bm25_index is None:
                print("  ⚠️  Warning: bm25_index not built, attempting to build...")
                self.demonstration_sampler.build_bm25_index(calibration_data, use_cache=False)
        
        try:
            delta_z_cache = self.task_vector_builder.compute_delta_z_with_centering(
                calibration_data,
                knn_index
            )
        finally:
            # Restore original selection_mode
            if icl_method != original_selection_mode:
                self.demonstration_sampler.set_selection_mode(original_selection_mode)
                print(f"  Restoring ICL selection mode: {icl_method} -> {original_selection_mode}")
        
        return delta_z_cache
    
    # calibrate_mu_with_cache deprecated (lambda is manual hyperparameter in paper, not auto-calibrated)
    
    def run_full_experiment(
        self,
        eval_baseline: bool = True
    ) -> Dict:
        """
        Run full experiment (baseline evaluation only)
        
        Note: This method no longer auto-calibrates lambda values or runs DeCoVec evaluation.
        To test DeCoVec, use --mode test_scale with --lambda_values specified manually.
        
        Args:
            eval_baseline: Whether to evaluate baseline
        
        Returns:
            Evaluation results for all methods
        """
        print("\n" + "=" * 80)
        print("Full Experiment Mode (Baseline Evaluation)")
        print("=" * 80)
        print(f"Experiment name: {self.config.experiment_name}_ICL")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"ICL Configuration: {self.n_shot}-shot")
        print("=" * 80)
        
        # 1. Load model
        self.setup_models()
        
        # 2. Load data
        calibration_data, test_data = self.data_loader.load_splits()
        
        # 3. Build kNN index
        embeddings, knn_index = self.build_knn_index(calibration_data)
        
        # 4. Precompute test set neighbors
        print("\n" + ">" * 80)
        print("Precomputing test set neighbors (for faster evaluation)")
        print(">" * 80)
        precomputed_neighbors = self.precompute_test_neighbors(test_data, calibration_data)
        
        # 5. Run experiments
        results = {}
        
        # Evaluate baselines
        max_samples = self.config.max_samples
        
        if eval_baseline:
            print("\n" + ">" * 80)
            results["zero_shot"] = self.baseline_evaluator.evaluate_zero_shot(
                test_data,
                max_samples=max_samples
            )
            
            print("\n" + ">" * 80)
            results["random_icl"] = self.baseline_evaluator.evaluate_random_icl(
                test_data,
                calibration_data,
                max_samples=max_samples
            )
            
            print("\n" + ">" * 80)
            results["icl"] = self.baseline_evaluator.evaluate_icl(
                test_data,
                calibration_data,
                precomputed_neighbors,
                max_samples=max_samples
            )
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_summary(results)
        
        print("\n" + "=" * 80)
        print("Experiment completed")
        print("=" * 80)
        
        return results
    
    def save_results(self, results: Dict):
        """Save experiment results"""
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            self.config.results_dir,
            f"results_icl_{timestamp}.json"
        )
        
        output = {
            "experiment_name": self.config.experiment_name + "_ICL",
            "timestamp": timestamp,
            "config": {
                "model": self.config.model_name,
                "n_shot": self.n_shot,
                "emb_model": self.emb_model_identifier,
                "dataset": self.dataset
            },
            "results": results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    def print_summary(self, results: Dict):
        """Print experiment summary"""
        print("\n" + "=" * 80)
        print("Experiment Summary (DeCoVec)")
        print("=" * 80)
        
        # Display different metrics based on dataset type
        print("\n[Full Results Comparison]")
        print(f"{'Method':<30} {'MC1':<10} {'MC2':<10} {'MC3':<10}")
        print("-" * 60)
        
        # Define display order
        method_order = [
            ("zero_shot", "Zero-shot (baseline)"),
            ("random_icl", "Random-ICL (baseline)"),
            ("icl", "ICL-KATE (baseline)"),
            ("icl_svd", "DeCoVec")
        ]
        
        for key, display_name in method_order:
            if key in results:
                mc1 = results[key].get("MC1", 0)
                mc2 = results[key].get("MC2", 0)
                mc3 = results[key].get("MC3", 0)
                print(f"{display_name:<30} {mc1:<10.2f} {mc2:<10.2f} {mc3:<10.2f}")
        
        print("\n" + "=" * 80)


