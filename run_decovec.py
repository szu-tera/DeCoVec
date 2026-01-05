"""
DeCoVec experiment entry script (refactored)
Uses a modular architecture for clarity and extensibility
"""
import os
import argparse
from decovec.experiment_manager import ExperimentManager
from simple_config import get_config, set_config, SimpleConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DeCoVec experiment (refactored)")
    
    # Core arguments
    parser.add_argument(
        "--mode",
                        choices=["full", "zero_shot", "random_icl", "icl", "test_scale"],
                        default="full",
        help="Experiment mode"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="truthfulqa",
        choices=[
            "truthfulqa",
            "truthfulqa_pool_ablation",
            "math500",
            "math500_pool_ablation",
            "aqua_rat",
            "aqua_rat_pool_ablation"
        ],
        help="Dataset selection (paper experiments)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            # Qwen2 family
            "qwen2-0.5b",
            "qwen2-1.5b",
            "qwen2-7b",        # Primary model in the paper
            # Other models
            "yi-6b",
            "llama2-7b",
            "llama3-8b",
            "gemma2-9b",
        ],
        help="Model selection (default: qwen2-7b from simple_config)"
    )
    
    # Experiment configuration
    parser.add_argument("--eval_baseline", action="store_true", default=True,
                        help="Whether to evaluate baselines (zero-shot and ICL)")
    parser.add_argument("--n_shot", type=int, default=None,
                        help="Number of ICL examples (default: generative tasks and news_factor=10, others=15). Supported in random_icl, icl, test_mu")
    parser.add_argument(
        "--example_order",
        type=str,
        default="ordered",
        choices=["ordered", "reverse", "random"],
        help="ICL example order (ordered=similarity order, reverse=reverse order, random=shuffle)"
    )
    parser.add_argument(
        "--example_order_seed",
        type=int,
        default=None,
        help="Random seed for example shuffling (used only when example_order=random)"
    )
    

    
    # Evaluation configuration
    parser.add_argument("--fast_mode", action="store_true",
                        help="Fast mode: evaluate only the first 100 samples")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of evaluation samples")
    parser.add_argument(
        "--calibration_pool_size",
        type=int,
        default=None,
        help="Limit the number of calibration samples in the pool (default uses all)"
    )
    parser.add_argument(
        "--calibration_pool_seed",
        type=int,
        default=None,
        help="Random seed before truncating the pool (default: no shuffle)"
    )
    
    parser.add_argument(
        "--icl_methods",
        type=str,
        default="kate",
        help="ICL strategy for SVD/mu tests (comma separated: kate,random_icl,bm25,mapping_error,topk)"
    )
    parser.add_argument(
        "--vector_icl_method",
        type=str,
        default=None,
        help="ICL strategy for building task vectors (default: first value of icl_methods). Decoupled from steer_icl_method to use different methods for vector building and inference"
    )
    parser.add_argument(
        "--baseline_icl_method",
        type=str,
        default=None,
        help="ICL method for the baseline prompt when computing delta_z (default None uses zero-shot; options: kate,random_icl,bm25,mapping_error,topk). Works with vector_icl_method to decouple baseline and ICL for delta_z"
    )
    
    # Lambda sweep configuration
    parser.add_argument(
        "--lambda_values",
        type=str,
        default=None,
        help="Lambda values to test (comma separated, e.g., 0.5,1.0,1.5,2.0)"
    )
    
    # Generation configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (for generative tasks; default from config.py)"
    )
    
    # Output saving configuration
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Save model outputs for generative datasets to CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/case",
        help="Output directory (default: results/case)"
    )
    
    # Self-consistency configuration
    parser.add_argument(
        "--self_consistency",
        action="store_true",
        help="Enable self-consistency mode (samples with temperature=0.7 and saves JSON results)"
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Run identifier for self-consistency output naming (e.g., result_1.json)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load global configuration
    config = get_config()
    
    # Override temperature if provided
    if args.temperature is not None:
        config.temperature = args.temperature
        print(f"[OK] Temperature: {args.temperature}")
    
    # Override model if provided
    if args.model is not None:
        model_configs = {
            # Qwen2 family
            "qwen2-0.5b": {
                "model_name": "Qwen/Qwen2-0.5B",
                "model_path": "checkpoints/qwen/Qwen2-0.5B"
            },
            "qwen2-1.5b": {
                "model_name": "Qwen/Qwen2-1.5B",
                "model_path": "checkpoints/qwen/Qwen2-1.5B"
            },
            "qwen2-7b": {
                "model_name": "Qwen/Qwen2-7B",
                "model_path": "checkpoints/qwen/Qwen2-7B"
            },
            # Other models
            "yi-6b": {
                "model_name": "01-ai/Yi-6B",
                "model_path": "checkpoints/yi/Yi-6B"
            },
            "llama2-7b": {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "model_path": "checkpoints/llama/Llama-2-7b-hf"
            },
            "llama3-8b": {
                "model_name": "meta-llama/Meta-Llama-3-8B",
                "model_path": "checkpoints/llama/Meta-Llama-3-8B"
            },
            "gemma2-9b": {
                "model_name": "google/gemma-2-9b",
                "model_path": "checkpoints/gemma/gemma-2-9b"
            },
        }
        
        if args.model in model_configs:
            model_cfg = model_configs[args.model]
            config.model_name = model_cfg["model_name"]
            config.model_path = model_cfg["model_path"]
            print(f"[OK] Model: {args.model} ({model_cfg['model_name']})")
        else:
            print(f"[ERR] Unknown model: {args.model}")
            return
    
    # Build experiment manager
    icl_methods_arg = args.icl_methods.split(',') if args.icl_methods else ["kate"]
    icl_methods = [m.strip() for m in icl_methods_arg if m.strip()]
    if not icl_methods:
        icl_methods = ["kate"]
    valid_icl_methods = {"kate", "random_icl", "bm25", "mapping_error", "topk"}
    invalid_icl = [m for m in icl_methods if m not in valid_icl_methods]
    if invalid_icl:
        print(f"[ERR] Invalid ICL method(s): {', '.join(invalid_icl)}")
        print("  Options: kate, random_icl, bm25, mapping_error, topk")
        return
    
    # Validate generative dataset flags
    GENERATIVE_DATASETS = {
        "math500",
        "math500_pool_ablation",
        "aqua_rat",
        "aqua_rat_pool_ablation",
    }
    if args.save_outputs and args.dataset not in GENERATIVE_DATASETS:
        print("[WARN] --save_outputs only applies to generative datasets")
        print(f"  Dataset '{args.dataset}' is not generative; ignoring --save_outputs")
        args.save_outputs = False
    
    # Validate self-consistency mode
    if args.self_consistency:
        if args.dataset not in GENERATIVE_DATASETS:
            print("[WARN] --self_consistency only applies to generative datasets")
            print(f"  Dataset '{args.dataset}' is not generative; ignoring --self_consistency")
            args.self_consistency = False
        else:
            print("[OK] Self-consistency enabled (temperature=0.7 sampling)")
            if args.run_id is not None:
                print(f"  Run ID: {args.run_id}")
            else:
                print("  [WARN] --run_id not set; output files will not include a run identifier")
    
    if args.save_outputs:
        print(f"[OK] Output saving enabled; results will be written to: {args.output_dir}")
    
    manager = ExperimentManager(
        dataset=args.dataset,
        n_shot=args.n_shot,
        icl_method=icl_methods[0],
        save_outputs=args.save_outputs,
        output_dir=args.output_dir,
        self_consistency=args.self_consistency,
        run_id=args.run_id,
        example_order=args.example_order,
        example_order_seed=args.example_order_seed
    )

    calibration_pool_size = args.calibration_pool_size
    if calibration_pool_size is not None and calibration_pool_size <= 0:
        print("[WARN] --calibration_pool_size must be positive; ignoring this argument")
        calibration_pool_size = None
    manager.data_loader.calibration_pool_size = calibration_pool_size
    if calibration_pool_size is not None:
        print(f"[OK] Calibration pool size limited to: {calibration_pool_size}")
    if args.calibration_pool_seed is not None:
        manager.data_loader.calibration_pool_seed = args.calibration_pool_seed
        print(f"[OK] Calibration pool seed: {args.calibration_pool_seed}")
    
    # Run experiment by mode
    if args.mode == "full":
        # Full experiment
        manager.run_full_experiment(eval_baseline=args.eval_baseline)
    
    elif args.mode == "zero_shot":
        # Evaluate zero-shot baseline only
        manager.setup_models()
        calibration_data, test_data = manager.data_loader.load_splits()
        
        max_samples = 100 if args.fast_mode else args.max_samples
        manager.baseline_evaluator.evaluate_zero_shot(test_data, max_samples=max_samples)
    
    elif args.mode == "random_icl":
        # Evaluate random ICL baseline only
        print(f"\n[OK] Using ICL example count: {manager.n_shot}")
        manager.setup_models()
        calibration_data, test_data = manager.data_loader.load_splits()
        
        max_samples = 100 if args.fast_mode else args.max_samples
        manager.baseline_evaluator.evaluate_random_icl(
            test_data,
            calibration_data,
            max_samples=max_samples
        )
    
    elif args.mode == "icl":
        # Evaluate ICL baselines (supports kate, bm25, random_icl)
        print(f"\n[OK] Using ICL example count: {manager.n_shot}")
        manager.setup_models()
        calibration_data, test_data = manager.data_loader.load_splits()
        
        max_samples = 100 if args.fast_mode else args.max_samples
        
        for icl_method in icl_methods:
            print(f"\n--- ICL method: {icl_method} ---")
            manager.set_icl_method(icl_method)
            
            # Build the index (kNN or BM25) for each ICL method
            embeddings, knn_index = manager.build_knn_index(calibration_data)
            
            # Precompute test neighbors using the current ICL method
            precomputed_neighbors = manager.precompute_test_neighbors(test_data, calibration_data)
            
            metrics = manager.baseline_evaluator.evaluate_icl(
                test_data,
                calibration_data,
                precomputed_neighbors,
                max_samples=max_samples
            )
            
            print(f"\nResults ({icl_method}):")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value}")
    
    elif args.mode == "test_scale":
        # Test different lambda scaling factors
        if args.lambda_values is None:
            print("[ERR] --mode test_scale requires --lambda_values")
            print("Example: python run_decovec.py --mode test_scale --lambda_values 0.5,1.0,1.5,2.0 --use_cache")
            return
        
        # Parse lambda values
        try:
            lambda_values = [float(x.strip()) for x in args.lambda_values.split(',')]
        except ValueError:
            print(f"[ERR] Invalid lambda value format: {args.lambda_values}")
            print("Example: python run_decovec.py --mode test_scale --lambda_values 0.5,1.0,1.5,2.0")
            return
        
        print(f"[OK] Lambda values: {lambda_values}")
        print(f"[OK] Using ICL example count: {manager.n_shot}")
        
        # Load models
        manager.setup_models()
        calibration_data, test_data = manager.data_loader.load_splits()
        
        max_samples = 100 if args.fast_mode else args.max_samples
        
        # Determine vector_icl_method (for task vector construction)
        vector_icl_method = args.vector_icl_method
        if vector_icl_method is None:
            # Default to first icl_method for backward compatibility
            vector_icl_method = icl_methods[0]
        else:
            # Validate vector_icl_method
            valid_icl_methods = {"kate", "random_icl", "bm25", "mapping_error", "topk"}
            if vector_icl_method not in valid_icl_methods:
                print(f"[ERR] Invalid vector_icl_method: {vector_icl_method}")
                print(f"  Options: {', '.join(valid_icl_methods)}")
                return
        
        # Determine baseline_icl_method (baseline prompt for delta_z)
        baseline_icl_method = args.baseline_icl_method
        if baseline_icl_method is not None:
            # Validate baseline_icl_method
            valid_icl_methods = {"kate", "random_icl", "bm25", "mapping_error", "topk"}
            if baseline_icl_method not in valid_icl_methods:
                print(f"[ERR] Invalid baseline_icl_method: {baseline_icl_method}")
                print(f"  Options: {', '.join(valid_icl_methods)}")
                return
        
        for icl_method in icl_methods:
            steer_icl_method = icl_method  # ICL method for inference
            
            # Use vector_icl_method if provided; otherwise, use steer_icl_method
            actual_vector_method = vector_icl_method if args.vector_icl_method is not None else steer_icl_method
            
            print(f"\n--- ICL method: {steer_icl_method} ---")
            if actual_vector_method != steer_icl_method:
                print(f"  steer_icl_method: {steer_icl_method} (used for inference steering)")
                print(f"  vector_icl_method: {actual_vector_method} (used for task vector construction)")
            if baseline_icl_method is not None:
                print(f"  baseline_icl_method: {baseline_icl_method} (baseline prompt for computing delta_z)")
            
            manager.set_icl_method(steer_icl_method)
            
            # Build the index (kNN or BM25) for the current ICL method used in inference
            embeddings, knn_index = manager.build_knn_index(calibration_data)
            
            # Precompute test neighbors using the current ICL method (steer_icl_method)
            precomputed_neighbors = manager.precompute_test_neighbors(test_data, calibration_data)
            
            # Compute delta_z using actual_vector_method
            delta_z_cache = manager.compute_delta_z_with_cache(
                calibration_data,
                knn_index,
                icl_method=actual_vector_method  # Use vector_icl_method to build task vectors
            )
            
            # Configure scale_tester methods
            manager.scale_tester.vector_icl_method = actual_vector_method
            manager.scale_tester.steer_icl_method = steer_icl_method
            manager.scale_tester.baseline_icl_method = baseline_icl_method
            
            results = manager.scale_tester.test_lambda_values(
                lambda_values=lambda_values,
                test_data=test_data,
                calibration_data=calibration_data,
                delta_z_cache=delta_z_cache,
                max_samples=max_samples
            )
            
            if actual_vector_method != steer_icl_method:
                print(f"\n[OK] Lambda sweep complete (steer_icl_method: {steer_icl_method}, vector_icl_method: {actual_vector_method})")
            else:
                print(f"\n[OK] Lambda sweep complete (ICL method: {steer_icl_method})")
    



if __name__ == "__main__":
    main()

