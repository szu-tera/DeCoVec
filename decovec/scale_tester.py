"""
Scaling Factor (λ) Testing Module
For quickly testing the effect of different scaling factors λ on results
Based on paper Equation (10): \\tilde{z}^t = z_{de}^t + \\lambda \\cdot v_{T}^t
"""
import os
import csv
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from decovec.demonstration_sampler import DemonstrationSampler
from decovec.decovec_processor import DeCoVecLogitsProcessor
from evaluate.evaluate_utils import (
    tokenize_prompt_and_continuation,
    create_question_stopping_criteria,
    get_dataset_stop_string
)


class ScaleTester:
    """Scaling Factor λ Tester
    
    Based on the DeCoVec method, tests the effect of different scaling factors λ on model performance.
    Task-Oriented Logits: \\tilde{z}^t = z_{de}^t + \\lambda \\cdot v_{T}^t
    """
    
    BINARY_DATASETS = set()  # Binary classification datasets not used in the paper
    GENERATIVE_DATASETS = {
        "math500",
        "math500_pool_ablation",
        "aqua_rat",
        "aqua_rat_pool_ablation",
    }
    
    # Dataset field mapping configuration
    # Defines which fields to use for question, full answer, and final answer for each dataset
    DATASET_FIELD_MAPPING = {
        "math500": {
            "question": ["question"],
            "full_answer": ["solution"],
            "final_answer": ["final_answer"]
        },
        "math500_pool_ablation": {
            "question": ["question"],
            "full_answer": ["solution"],
            "final_answer": ["final_answer"]
        },
        "aqua_rat": {
            "question": ["question"],
            "full_answer": ["answer"],
            "final_answer": ["final_answer"]
        },
        "aqua_rat_pool_ablation": {
            "question": ["question"],
            "full_answer": ["answer"],
            "final_answer": ["final_answer"]
        }
    }

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        demonstration_sampler,
        task_vector_builder,
        dataset: str = "truthfulqa",
        cache_manager = None,
        save_outputs: bool = False,
        output_dir: str = "results/case",
        model_name: str = None,
        self_consistency: bool = False,
        run_id: int = None,
        icl_method: str = "kate"
    ):
        """
        Initialize Scaling Factor λ Tester
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            device: Device
            demonstration_sampler: ICL example selector
            task_vector_builder: DeCoVec computer
            dataset: Dataset type
            cache_manager: Cache manager
            save_outputs: Whether to save model outputs for generative datasets
            output_dir: Output file save directory
            model_name: Model name (for filename generation)
            self_consistency: Whether to enable self-consistency mode (uses temperature=0.7 sampling when enabled)
            run_id: Run ID (for filename, e.g., result_1.json)
            icl_method: ICL method (for filename, to avoid overwriting results from different methods)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.demonstration_sampler = demonstration_sampler
        self.task_vector_builder = task_vector_builder
        self.dataset = dataset
        self.cache_manager = cache_manager
        self.save_outputs = save_outputs
        self.output_dir = output_dir
        self.model_name = self._normalize_model_name(model_name) if model_name else "unknown_model"
        self.self_consistency = self_consistency
        self.run_id = run_id
        self.icl_method = icl_method  # For filename (backward compatibility)
        # Record n_shot (for distinguishing output files of different shots and building logits cache)
        try:
            self.n_shot = int(getattr(self.demonstration_sampler, "n_shot"))
        except Exception:
            self.n_shot = None
        self._delta_z_warning_issued = False
        
        # Get steering_computer from task_vector_builder
        self.steering_computer = task_vector_builder.steering_computer if task_vector_builder else None
        
        # Support decoupled ICL methods
        # vector_icl_method: ICL method used to construct task vectors
        # steer_icl_method: ICL method used for steering during inference (current demonstration_sampler mode)
        # baseline_icl_method: ICL method used for baseline prompt when computing δz (None means use zero-shot)
        self.vector_icl_method = None  # Will be set at runtime
        self.steer_icl_method = None  # Will be set at runtime
        self.baseline_icl_method = None  # Will be set at runtime (None means use zero-shot)
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to be suitable for filenames"""
        if not model_name:
            return "unknown_model"
        # Replace characters unsuitable for filenames
        normalized = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        # Extract the last part (usually the model identifier)
        parts = normalized.split("_")
        # Keep meaningful parts
        if len(parts) > 1:
            # e.g., "Qwen_Qwen2.5-7B-Instruct" -> "Qwen2.5-7B-Instruct"
            # or "meta_llama_Meta-Llama-3.1-8B" -> "Meta-Llama-3.1-8B"
            return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        return normalized
    
    def _get_field_value(self, item: Dict, field_type: str) -> str:
        """
        Get field value based on dataset mapping configuration
        
        Args:
            item: Data item
            field_type: Field type ("question", "full_answer", "final_answer")
        
        Returns:
            field_value: Field value
        """
        # Get field mapping for current dataset
        mapping = self.DATASET_FIELD_MAPPING.get(self.dataset, {})
        field_names = mapping.get(field_type, [])
        
        # Try to get field value by priority
        for field_name in field_names:
            if field_name in item and item[field_name]:
                return item[field_name]
        
        return ""
    
    def _get_question(self, item: Dict) -> str:
        """
        Unified method to get question text
        
        Args:
            item: Data item
        
        # Record n_shot (for distinguishing output files of different n_shots, to avoid overwriting)
        try:
            self.n_shot = int(getattr(demonstration_sampler, "n_shot"))
        except Exception:
            self.n_shot = None
        Returns:
            question: Question text
        """
        # First try to use mapping configuration
        question = self._get_field_value(item, "question")
        if question:
            return question
        
        # Fall back to common fields (for compatibility)
        return item.get("problem") or item.get("question", "")
    
    def _get_full_answer(self, item: Dict) -> str:
        """
        Unified method to get full answer (complete response/solution process)
        
        Args:
            item: Data item
        
        Returns:
            full_answer: Full answer text
        """
        return self._get_field_value(item, "full_answer")
    
    def _get_final_answer(self, item: Dict) -> str:
        """
        Unified method to get final answer
        
        Args:
            item: Data item
        
        Returns:
            final_answer: Final answer
        """
        return self._get_field_value(item, "final_answer")
    
    def _is_multiple_choice_dataset(self) -> bool:
        """Check if dataset is a classification/multiple-choice dataset that can cache logits"""
        return self.dataset in {"truthfulqa", "truthfulqa_pool_ablation"}
    
    def _is_generative_dataset(self) -> bool:
        """Check if dataset is a generative dataset (requires real-time generation)"""
        return self.dataset in self.GENERATIVE_DATASETS

    def _dataset_alias(self) -> str:
        """Map ablation datasets back to base dataset names for prompt/evaluator compatibility."""
        alias_map = {
            "truthfulqa_pool_ablation": "truthfulqa",
            "math500_pool_ablation": "math500",
            "aqua_rat_pool_ablation": "aqua_rat",
        }
        return alias_map.get(self.dataset, self.dataset)
    
    def _get_sample_key(self, item: Dict, idx: int) -> str:
        """Generate stable sample key (prefer explicit id, otherwise use question/prompt)"""
        item = item or {}
        for key in ["id", "uuid", "question_id"]:
            if key in item:
                return str(item[key])
        if "question" in item:
            return str(item["question"])
        if "prompt" in item:
            return str(item["prompt"])
        return f"sample_{idx}"
    
    def _store_choice_delta_z(
        self,
        sample_delta_store: Dict,
        sample_key: str,
        section: str,
        choice: str,
        delta_z_list: List[torch.Tensor]
    ):
        """Cache δz for a specific sample's section + choice (store as CPU tensors for persistence)"""
        sample_entry = sample_delta_store.setdefault(sample_key, {})
        section_entry = sample_entry.setdefault(section, {})
        section_entry[choice] = [dz.cpu() for dz in delta_z_list]
    
    def _get_choice_delta_z(
        self,
        sample_delta_store: Dict,
        sample_key: str,
        section: str,
        choice: str
    ) -> List[torch.Tensor]:
        """Get δz list for specified sample section/choice"""
        delta_z_list = (
            sample_delta_store
            .get(sample_key, {})
            .get(section, {})
            .get(choice)
        )
        if delta_z_list is None and not self._delta_z_warning_issued:
            print(f"⚠️  δz not found for sample {sample_key} {section}:{choice}, falling back to mean vector")
            self._delta_z_warning_issued = True
        return delta_z_list
    
    def test_lambda_values(
        self,
        lambda_values: List[float],
        test_data: List[Dict],
        calibration_data: List[Dict],
        delta_z_cache: Dict,
        max_samples: int = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Quickly test the effect of different λ values
        
        Args:
            lambda_values: List of λ values to test
            test_data: Test data
            calibration_data: Calibration set data
            delta_z_cache: δz cache
            max_samples: Maximum number of samples
        
        Returns:
            results: {lambda_value: metrics} dictionary
        """
        print("\n" + "=" * 80)
        print(f"Quick λ Value Testing")
        print("=" * 80)
        print(f"Testing λ values: {lambda_values}")
        if self.self_consistency:
            print(f"✓ Self-Consistency mode enabled (temperature=0.7, sampling)")
            if self.run_id is not None:
                print(f"  Run ID: {self.run_id}")
        
        # Build logits cache (compute once, reuse for all λ values)
        print("\nBuilding Logits cache...")
        n_shot = self.n_shot
        if n_shot is None:
            n_shot = getattr(self.demonstration_sampler, "n_shot", None)
        if n_shot is None:
            raise ValueError("ScaleTester requires demonstration_sampler.n_shot (currently None), cannot build logits cache")
        logits_cache = self.build_logits_cache(
            test_data,
            calibration_data,
            delta_z_cache,
            max_samples=max_samples,
            n_shot=n_shot
        )
        
        # Get delta_z_mean
        delta_z_mean = torch.tensor(delta_z_cache["delta_z_mean"]).to(self.device)
        
        # Test each λ value
        results = {}
        
        for lambda_val in lambda_values:
            print(f"\nTesting λ = {lambda_val:.4f}...")
            metrics = self.evaluate_with_logits_cache(
                logits_cache,
                delta_z_cache,
                delta_z_mean,
                lambda_val
            )
            results[lambda_val] = metrics
            
            # Print results
            if self._is_generative_dataset():
                print(f"  Accuracy: {metrics.get('accuracy', 0):.2f}%")
            else:
                print(f"  MC1: {metrics.get('MC1', 0):.2f}%")
                print(f"  MC2: {metrics.get('MC2', 0):.2f}%")
                print(f"  MC3: {metrics.get('MC3', 0):.2f}%")
        
        # Print comparison table
        self._print_comparison_table(results, lambda_values)
        
        return results
    
    def build_logits_cache(
        self,
        test_data: List[Dict],
        calibration_data: List[Dict],
        delta_z_cache: Dict,
        max_samples: int = None,
        n_shot: int = None
    ) -> Dict:
        """
        Build logits cache for test set
        
        For multiple-choice datasets: Cache complete logits (each token position for each candidate answer)
        For generative datasets: Only cache ICL prompt
        
        Args:
            test_data: Test data
            calibration_data: Calibration set data
            max_samples: Maximum number of samples
            n_shot: Number of ICL examples
        
        Returns:
            logits_cache: Cache containing all samples
        """
        eval_data = test_data[:max_samples] if max_samples else test_data
        
        # Try to load from disk cache
        sample_delta_store = delta_z_cache.setdefault("sample_delta_z", {})
        
        if max_samples:
            print(f"⚡ Fast mode: Processing first {len(eval_data)}/{len(test_data)} samples")
        
        # Choose different caching strategies based on dataset type
        if self._is_multiple_choice_dataset():
            print(f"  → Multiple-choice dataset: Caching complete logits (reusable for different λ values)")
            cache_data, sample_keys = self._build_multiple_choice_logits_cache(
                eval_data,
                calibration_data,
                sample_delta_store
            )
        else:
            print(f"  → Generative dataset: Only caching prompt (real-time generation each time)")
            cache_data, sample_keys = self._build_generative_cache(
                eval_data,
                calibration_data
            )
        
        if sample_keys:
            cache_data["sample_delta_z"] = {
                key: sample_delta_store.get(key, {})
                for key in sample_keys
            }
        
        return cache_data
    
    def _build_generative_cache(
        self,
        test_data: List[Dict],
        calibration_data: List[Dict]
    ) -> Tuple[Dict, List[str]]:
        """
        Build cache for generative datasets (only cache prompts)
        
        Args:
            test_data: Test data
            calibration_data: Calibration set data
        
        Returns:
            (cache data, sample key list)
        """
        cache_samples = []
        sample_keys = []
        
        print(f"Processing {len(test_data)} test samples...")
        
        dataset_for_prompt = self._dataset_alias()

        # Save current selection_mode
        original_selection_mode = self.demonstration_sampler.selection_mode
        
        # Determine ICL methods to use
        steer_icl_method = self.steer_icl_method if self.steer_icl_method is not None else original_selection_mode
        vector_icl_method = self.vector_icl_method if self.vector_icl_method is not None else original_selection_mode
        baseline_icl_method = self.baseline_icl_method  # None means use zero-shot
        
        # If vector_icl_method differs from current selector mode, need to temporarily switch
        use_vector_icl = (vector_icl_method != steer_icl_method)
        # If baseline_icl_method is not None, need to construct baseline ICL prompt
        use_baseline_icl = (baseline_icl_method is not None)
        
        for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Building cache"):
            sample_key = self._get_sample_key(item, idx)
            
            # Get ICL examples for inference (using steer_icl_method)
            if steer_icl_method != original_selection_mode:
                self.demonstration_sampler.set_selection_mode(steer_icl_method)
            
            examples_steer = self.demonstration_sampler.get_icl_examples(
                idx,
                calibration_data,
                use_precomputed=True
            )
            
            # Construct ICL prompt for inference (using steer_icl_method)
            icl_prompt = self.demonstration_sampler.construct_icl_prompt(examples_steer, item)
            
            # If using different vector_icl_method, get ICL examples for task vector construction
            vector_icl_prompt = None
            if use_vector_icl:
                if vector_icl_method != self.demonstration_sampler.selection_mode:
                    self.demonstration_sampler.set_selection_mode(vector_icl_method)
                
                examples_vector = self.demonstration_sampler.get_icl_examples(
                    idx,
                    calibration_data,
                    use_precomputed=True
                )
                
                # Construct ICL prompt for task vector construction (using vector_icl_method)
                vector_icl_prompt = self.demonstration_sampler.construct_icl_prompt(examples_vector, item)
            
            # If using baseline_icl_method, get baseline ICL examples for δz computation
            baseline_icl_prompt = None
            if use_baseline_icl:
                if baseline_icl_method != self.demonstration_sampler.selection_mode:
                    self.demonstration_sampler.set_selection_mode(baseline_icl_method)
                
                examples_baseline = self.demonstration_sampler.get_icl_examples(
                    idx,
                    calibration_data,
                    use_precomputed=True
                )
                
                # Construct baseline ICL prompt for δz computation (using baseline_icl_method)
                baseline_icl_prompt = self.demonstration_sampler.construct_icl_prompt(examples_baseline, item)
            
            # Restore original selection_mode (if changed)
            if steer_icl_method != original_selection_mode or (use_vector_icl and vector_icl_method != original_selection_mode) or (use_baseline_icl and baseline_icl_method != original_selection_mode):
                self.demonstration_sampler.set_selection_mode(original_selection_mode)
            
            sample_cache = {
                "icl_prompt": icl_prompt,
                "vector_icl_prompt": vector_icl_prompt,  # If None, use icl_prompt
                "baseline_icl_prompt": baseline_icl_prompt,  # If None, use zero-shot prompt
                "item": item,
                "sample_key": sample_key
            }
            
            cache_samples.append(sample_cache)
            sample_keys.append(sample_key)
        
        # Ensure original selection_mode is restored
        if self.demonstration_sampler.selection_mode != original_selection_mode:
            self.demonstration_sampler.set_selection_mode(original_selection_mode)
        
        return {"samples": cache_samples}, sample_keys
    
    def _build_multiple_choice_logits_cache(
        self,
        test_data: List[Dict],
        calibration_data: List[Dict],
        sample_delta_store: Dict
    ) -> Tuple[Dict, List[str]]:
        """
        Build logits cache for multiple-choice datasets
        
        For each candidate answer of each test sample, forward pass to get complete logits for continuation part
        
        Args:
            test_data: Test data
            calibration_data: Calibration set data
        
        Returns:
            (cache data, sample key list)
        """
        cache_samples = []
        sample_keys = []
        
        print(f"Processing {len(test_data)} test samples...")
        dataset_for_prompt = self._dataset_alias()
        base_dataset = dataset_for_prompt
        
        # Save current selection_mode
        original_selection_mode = self.demonstration_sampler.selection_mode
        
        # Determine ICL methods to use
        steer_icl_method = self.steer_icl_method if self.steer_icl_method is not None else original_selection_mode
        vector_icl_method = self.vector_icl_method if self.vector_icl_method is not None else original_selection_mode
        baseline_icl_method = self.baseline_icl_method  # None means use zero-shot
        
        # If vector_icl_method differs from current selector mode, need to temporarily switch
        use_vector_icl = (vector_icl_method != steer_icl_method)
        # If baseline_icl_method is not None, need to construct baseline ICL prompt
        use_baseline_icl = (baseline_icl_method is not None)
        
        for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Building logits cache"):
            sample_key = self._get_sample_key(item, idx)
            
            # Get ICL examples for inference (using steer_icl_method)
            if steer_icl_method != original_selection_mode:
                self.demonstration_sampler.set_selection_mode(steer_icl_method)
            
            examples_steer = self.demonstration_sampler.get_icl_examples(
                idx,
                calibration_data,
                use_precomputed=True
            )
            
            # Construct ICL prompt for inference (using steer_icl_method)
            icl_prompt = self.demonstration_sampler.construct_icl_prompt(examples_steer, item)
            
            # If using different vector_icl_method, get ICL examples for task vector construction
            vector_icl_prompt = None
            if use_vector_icl:
                if vector_icl_method != self.demonstration_sampler.selection_mode:
                    self.demonstration_sampler.set_selection_mode(vector_icl_method)
                
                examples_vector = self.demonstration_sampler.get_icl_examples(
                    idx,
                    calibration_data,
                    use_precomputed=True
                )
                
                # Construct ICL prompt for task vector construction (using vector_icl_method)
                vector_icl_prompt = self.demonstration_sampler.construct_icl_prompt(examples_vector, item)
            
            # If using baseline_icl_method, get baseline ICL examples for δz computation
            baseline_icl_prompt = None
            if use_baseline_icl:
                if baseline_icl_method != self.demonstration_sampler.selection_mode:
                    self.demonstration_sampler.set_selection_mode(baseline_icl_method)
                
                examples_baseline = self.demonstration_sampler.get_icl_examples(
                    idx,
                    calibration_data,
                    use_precomputed=True
                )
                
                # Construct baseline ICL prompt for δz computation (using baseline_icl_method)
                baseline_icl_prompt = self.demonstration_sampler.construct_icl_prompt(examples_baseline, item)
            
            # Restore original selection_mode (if changed)
            if steer_icl_method != original_selection_mode or (use_vector_icl and vector_icl_method != original_selection_mode) or (use_baseline_icl and baseline_icl_method != original_selection_mode):
                self.demonstration_sampler.set_selection_mode(original_selection_mode)
            
            # Construct baseline prompt (if baseline_icl_method is None, use zero-shot)
            if baseline_icl_prompt is not None:
                baseline_prompt = baseline_icl_prompt
            else:
                baseline_prompt = DemonstrationSampler.construct_zero_shot_prompt(
                    item,
                    dataset_type=dataset_for_prompt
                )
            
            zero_shot_prompt = DemonstrationSampler.construct_zero_shot_prompt(
                item,
                dataset_type=dataset_for_prompt
            )
            
            sample_cache = {
                "icl_prompt": icl_prompt,
                "item": item,
                "sample_key": sample_key
            }
            
            # Cache different choices based on dataset type
            if base_dataset == "truthfulqa":
                # TruthfulQA needs to cache all candidate answers for MC1 and MC2
                mc1_choices = item.get("mc1_choices", [])
                if isinstance(mc1_choices, str):
                    mc1_choices = eval(mc1_choices)
                
                mc2_choices = item.get("mc2_choices", [])
                if isinstance(mc2_choices, str):
                    mc2_choices = eval(mc2_choices)
                
                mc2_correct_choices = item.get("mc2_correct_choices", [])
                if isinstance(mc2_correct_choices, str):
                    mc2_correct_choices = eval(mc2_correct_choices)
                
                mc1_correct_idx = item.get("mc1_correct_idx", None)
                if mc1_correct_idx is not None:
                    mc1_correct_idx = int(mc1_correct_idx)
                
                # Save parsed choices info (ensure consistent order)
                sample_cache["mc1_choices"] = mc1_choices
                sample_cache["mc1_correct_idx"] = mc1_correct_idx
                sample_cache["mc2_choices"] = mc2_choices
                sample_cache["mc2_correct_choices"] = mc2_correct_choices
                
                sample_cache["mc1_choices_data"] = self._cache_choices_logits(
                    icl_prompt,
                    baseline_prompt,
                    mc1_choices,
                    sample_key,
                    sample_delta_store,
                    "mc1",
                    vector_icl_prompt=vector_icl_prompt
                )
                sample_cache["mc2_choices_data"] = self._cache_choices_logits(
                    icl_prompt,
                    baseline_prompt,
                    mc2_choices,
                    sample_key,
                    sample_delta_store,
                    "mc2",
                    vector_icl_prompt=vector_icl_prompt
                )
            
            cache_samples.append(sample_cache)
            sample_keys.append(sample_key)
        
        # Ensure original selection_mode is restored
        if self.demonstration_sampler.selection_mode != original_selection_mode:
            self.demonstration_sampler.set_selection_mode(original_selection_mode)
        
        return {"samples": cache_samples}, sample_keys
    
    def _cache_choices_logits(
        self,
        prompt: str,
        baseline_prompt: str,
        choices: List[str],
        sample_key: str,
        sample_delta_store: Dict,
        section: str,
        vector_icl_prompt: str = None
    ) -> List[Dict]:
        """
        Cache logits for given prompt and choices, and synchronize per-token δz caching
        Optimization: Reduce CPU-GPU communication overhead
        
        Args:
            prompt: ICL prompt (for inference, using steer_icl_method)
            baseline_prompt: Baseline prompt (for δz computation, may be zero-shot or baseline ICL prompt)
            choices: List of candidate answers
            sample_key: Current sample identifier
            sample_delta_store: Sample δz storage container
            section: Section name ("mc1"/"mc2" for TruthfulQA)
            vector_icl_prompt: ICL prompt for task vector construction (if None, use prompt)
        
        Returns:
            Cache data for each candidate answer (including token_ids and logits)
        """
        choices_data = []
        
        # If vector_icl_prompt is specified, use it to compute delta_z; otherwise use prompt
        # This allows using different ICL methods when constructing task vectors
        delta_z_prompt = vector_icl_prompt if vector_icl_prompt is not None else prompt
        
        for choice in choices:
            # Use lm-eval's tokenize method
            context_enc, continuation_enc = tokenize_prompt_and_continuation(
                self.tokenizer, prompt, choice
            )
            baseline_context_enc, _ = tokenize_prompt_and_continuation(
                self.tokenizer, baseline_prompt, choice
            )
            
            # Prompt for computing delta_z (if using different ICL methods)
            if vector_icl_prompt is not None:
                vector_context_enc, _ = tokenize_prompt_and_continuation(
                    self.tokenizer, vector_icl_prompt, choice
                )
            else:
                vector_context_enc = context_enc
            
            if len(continuation_enc) == 0:
                choices_data.append({
                    "choice": choice,
                    "token_ids": [],
                    "logits": []
                })
                self._store_choice_delta_z(
                    sample_delta_store,
                    sample_key,
                    section,
                    choice,
                    []
                )
                continue
            
            # Construct inputs
            inp = (context_enc + continuation_enc)[:-1]  # Prompt for inference
            baseline_inp = (baseline_context_enc + continuation_enc)[:-1]  # Baseline prompt for δz computation
            vector_inp = (vector_context_enc + continuation_enc)[:-1]  # Prompt for delta_z computation
            
            with torch.no_grad():
                # Logits for inference (using steer_icl_method prompt)
                input_ids = torch.tensor([inp]).to(self.device)
                outputs = self.model(input_ids)
                logits = outputs.logits[0]
                
                # Baseline logits (for δz computation, may be zero-shot or baseline ICL)
                baseline_input_ids = torch.tensor([baseline_inp]).to(self.device)
                baseline_outputs = self.model(baseline_input_ids)
                baseline_logits = baseline_outputs.logits[0]
                
                # Logits for delta_z computation (using vector_icl_method prompt)
                vector_input_ids = torch.tensor([vector_inp]).to(self.device)
                vector_outputs = self.model(vector_input_ids)
                vector_logits = vector_outputs.logits[0]
                
                # Select continuation part of logits
                inplen = len(inp)
                baseline_inplen = len(baseline_inp)
                vector_inplen = len(vector_inp)
                contlen = len(continuation_enc)
                cont_logits = logits[inplen - contlen : inplen]  # Logits for inference
                baseline_cont_logits = baseline_logits[baseline_inplen - contlen : baseline_inplen]  # Baseline logits for δz computation
                vector_cont_logits = vector_logits[vector_inplen - contlen : vector_inplen]  # Logits for delta_z computation
                
                # Compute all delta_z in batch on GPU, then move to CPU once
                # Note: Use vector_cont_logits and baseline_cont_logits to compute delta_z
                delta_z_list = []
                for i in range(len(continuation_enc)):
                    delta_z = self.task_vector_builder.steering_computer.compute_delta_z(
                        baseline_cont_logits[i],  # Use baseline prompt logits (may be zero-shot or baseline ICL)
                        vector_cont_logits[i]  # Use vector_icl_method logits
                    )
                    delta_z_list.append(delta_z)
                
                # Optimization: Move all logits and delta_z from GPU to CPU at once (reduce communication overhead)
                logits_list = [cont_logits[i].cpu() for i in range(len(continuation_enc))]
                delta_z_list_cpu = [dz.cpu() for dz in delta_z_list]
                
                self._store_choice_delta_z(
                    sample_delta_store,
                    sample_key,
                    section,
                    choice,
                    delta_z_list_cpu
                )
                
                choices_data.append({
                    "choice": choice,
                    "token_ids": continuation_enc,
                    "logits": logits_list
                })
        
        return choices_data
    
    def evaluate_with_logits_cache(
        self,
        logits_cache: Dict,
        delta_z_cache: Dict,
        delta_z_mean: torch.Tensor,
        lambda_scale: float
    ) -> Dict[str, float]:
        """
        Evaluate using logits cache
        
        Args:
            logits_cache: Pre-built logits cache
            delta_z_mean: Mean δz
            lambda_scale: Calibration strength
        
        Returns:
            metrics: Evaluation metrics
        """
        samples = logits_cache["samples"]
        sample_delta_store = delta_z_cache.get("sample_delta_z", {})
        
        # Choose evaluation method based on dataset type
        if self.dataset in self.GENERATIVE_DATASETS:
            return self._evaluate_generative_reasoning(samples, lambda_scale)
        else:
            # truthfulqa and truthfulqa_pool_ablation
            return self._evaluate_truthfulqa(samples, sample_delta_store, delta_z_mean, lambda_scale)
    
    def _evaluate_generative_reasoning(
        self,
        samples: List[Dict],
        lambda_scale: float
    ) -> Dict[str, float]:
        """Evaluate generative reasoning tasks (including mathematical and commonsense reasoning)"""
        if self.steering_computer is None:
            raise ValueError("Steering computer not initialized, cannot perform dynamic SVD evaluation")
        
        base_dataset = self._dataset_alias()
        dataset_to_evaluator = {
            "math500": ("evaluate.evaluate_math500", "Math500Evaluator"),
            "aqua_rat": ("evaluate.evaluate_aqua_rat", "AquaRATEvaluator"),
        }
        module_info = dataset_to_evaluator.get(base_dataset)
        if module_info is None:
            raise ValueError(f"Unsupported generative dataset: {self.dataset}")
        
        module = __import__(module_info[0], fromlist=[module_info[1]])
        evaluator_cls = getattr(module, module_info[1])
        evaluator = evaluator_cls(self.tokenizer, self.device, model_name=self.model_name)
        
        correct_count = 0
        total_count = 0
        stop_marker = get_dataset_stop_string(base_dataset)
        
        # Prepare data for saving outputs
        output_rows = []  # CSV format (original logic)
        self_consistency_results = []  # JSON format (self-consistency mode)
        
        for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating {self.dataset} (lambda={lambda_scale:.4f})")):
            item = sample["item"]
            prompt = sample["icl_prompt"]  # ICL prompt for inference (using steer_icl_method)
            vector_icl_prompt = sample.get("vector_icl_prompt")  # ICL prompt for task vector construction (using vector_icl_method)
            baseline_icl_prompt = sample.get("baseline_icl_prompt")  # Baseline ICL prompt for δz computation (if None, use zero-shot)
            
            # If baseline_icl_prompt is None, use zero-shot prompt
            if baseline_icl_prompt is None:
                baseline_prompt = DemonstrationSampler.construct_zero_shot_prompt(
                    item,
                    dataset_type=base_dataset
                )
            else:
                baseline_prompt = baseline_icl_prompt
            
            svd_processor = DeCoVecLogitsProcessor(
                model=self.model,
                tokenizer=self.tokenizer,
                zero_shot_prompt=baseline_prompt,  # Pass baseline prompt (may be zero-shot or baseline ICL)
                icl_prompt=prompt,
                lambda_scale=lambda_scale,
                steering_computer=self.steering_computer,
                device=self.device,
                vector_icl_prompt=vector_icl_prompt  # Pass ICL prompt for task vector construction
            )
            
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids)
            input_length = input_ids.shape[1]
            
            # Create stopping criteria: dynamically detect stop marker based on dataset
            stopping_criteria = create_question_stopping_criteria(
                tokenizer=self.tokenizer,
                dataset_name=base_dataset,
                initial_input_length=input_length
            )
            
            # Determine sampling parameters based on self_consistency mode
            if self.self_consistency:
                do_sample = True
                temperature = 0.7
            else:
                do_sample = False
                temperature = None  # Greedy decoding doesn't need temperature
            
            with torch.no_grad():
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": 1024,  # Output max 1024 tokens
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "logits_processor": [svd_processor],
                    "stopping_criteria": stopping_criteria
                }
                if do_sample:
                    generate_kwargs["temperature"] = temperature
                
                output_ids = self.model.generate(**generate_kwargs)
            
            generated_tokens = output_ids[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Save original generated text (for printing)
            original_generated_text = generated_text
            
            # Also filter content after stop marker during decoding (double insurance)
            if stop_marker and stop_marker in generated_text:
                generated_text = generated_text.split(stop_marker)[0].strip()
            
            # Unified method to get final answer as ground_truth
            ground_truth = self._get_final_answer(item)
            # If not found in mapping configuration, use fallback logic
            if not ground_truth:
                ground_truth = item.get("final_answer") or item.get("answer", "")
            
            # Unified field retrieval (using mapping configuration)
            question = self._get_question(item)
            full_answer = self._get_full_answer(item)
            final_answer = self._get_final_answer(item)
            
            # If final answer not found in mapping configuration, use ground_truth as fallback
            if not final_answer:
                final_answer = ground_truth
            
            try:
                predicted_answer = evaluator.extract_final_answer(generated_text)
                is_correct = evaluator.check_correctness(predicted_answer, ground_truth)
                
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                # If saving outputs, collect data
                if self.save_outputs:
                    output_rows.append({
                        "question": question,
                        "full_answer": full_answer,
                        "final_answer": final_answer,
                        "model_output": generated_text,
                        "extracted_answer": predicted_answer
                    })
                
                # Self-consistency mode: Save simplified JSON format (only final_answer and ground_truth)
                if self.self_consistency:
                    self_consistency_results.append({
                        "sample_id": idx,
                        "predicted_answer": predicted_answer,
                        "ground_truth": ground_truth,
                        "is_correct": is_correct
                    })
            except Exception as e:
                total_count += 1
                # Save output even on error (if any)
                if self.save_outputs:
                    output_rows.append({
                        "question": question,
                        "full_answer": full_answer,
                        "final_answer": final_answer,
                        "model_output": generated_text,
                        "extracted_answer": ""
                    })
                
                # Self-consistency mode: Record even on error
                if self.self_consistency:
                    self_consistency_results.append({
                        "sample_id": idx,
                        "predicted_answer": "",
                        "ground_truth": ground_truth,
                        "is_correct": False
                    })
        
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
        
        # If saving outputs, write to CSV file (original logic, non self-consistency mode)
        if self.save_outputs and output_rows and not self.self_consistency:
            os.makedirs(self.output_dir, exist_ok=True)
            # Format λ value as filename-friendly format (replace decimal point with underscore)
            lambda_str = str(lambda_scale).replace(".", "_").replace("-", "neg")
            shot_str = f"shot{self.n_shot}_" if self.n_shot is not None else ""
            filename = f"{self.dataset}_test_lambda_{lambda_str}_{shot_str}{self.icl_method}_{self.model_name}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["question", "full_answer", "final_answer", "model_output", "extracted_answer"])
                writer.writeheader()
                writer.writerows(output_rows)
            
            print(f"\n✓ Saved model outputs to: {filepath}")
            print(f"  Saved {len(output_rows)} records total")
        
        # Self-consistency mode: Save JSON format (only final_answer and ground_truth)
        if self.self_consistency and self_consistency_results:
            os.makedirs(self.output_dir, exist_ok=True)
            # Format λ value as filename-friendly format
            lambda_str = str(lambda_scale).replace(".", "_").replace("-", "neg")
            shot_str = f"shot{self.n_shot}_" if self.n_shot is not None else ""
            # Filename: include run_id (if provided) and icl_method
            if self.run_id is not None:
                filename = f"{self.dataset}_self_consistency_lambda_{lambda_str}_{shot_str}{self.icl_method}_run_{self.run_id}_{self.model_name}.json"
            else:
                filename = f"{self.dataset}_self_consistency_lambda_{lambda_str}_{shot_str}{self.icl_method}_{self.model_name}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self_consistency_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Saved Self-Consistency results to: {filepath}")
            print(f"  Saved {len(self_consistency_results)} records total")
        
        return {"accuracy": accuracy}
    
    def _evaluate_truthfulqa(
        self,
        samples: List[Dict],
        sample_delta_store: Dict,
        delta_z_mean: torch.Tensor,
        lambda_scale: float
    ) -> Dict[str, float]:
        """Evaluate TruthfulQA dataset (using cached logits)"""
        from evaluate.evaluate_truthfulqa import QAEvaluator

        base_dataset = self._dataset_alias()
        evaluator = QAEvaluator(self.tokenizer, self.device, dataset_name=base_dataset)
        mc1_scores = []
        mc2_scores = []
        mc3_scores = []
        
        for idx, sample in tqdm(
            enumerate(samples),
            total=len(samples),
            desc=f"Evaluating TruthfulQA (lambda={lambda_scale:.4f})"
        ):
            # Read parsed choices from cache (ensure consistent order)
            mc1_choices = sample.get("mc1_choices", [])
            mc1_correct_idx = sample.get("mc1_correct_idx", None)
            mc2_choices = sample.get("mc2_choices", [])
            mc2_correct_choices = sample.get("mc2_correct_choices", [])
            sample_key = sample.get("sample_key") or self._get_sample_key(sample.get("item", {}), idx)
            
            # Compute MC1 (using cached logits, batch optimized)
            if mc1_choices and mc1_correct_idx is not None:
                mc1_choices_data = sample.get("mc1_choices_data", [])
                
                # Batch get delta_z and logits for all choices
                delta_z_lists = []
                ft_logits_lists = []
                token_ids_lists = []
                
                for choice_data in mc1_choices_data:
                    cached_logits = choice_data["logits"]
                    token_ids = choice_data["token_ids"]
                    choice = choice_data["choice"]
                    
                    delta_z_list = self._get_choice_delta_z(
                        sample_delta_store,
                        sample_key,
                        "mc1",
                        choice
                    )
                    
                    # If no delta_z, use mean
                    if delta_z_list is None or len(delta_z_list) == 0:
                        delta_z_list = [delta_z_mean] * len(token_ids)
                    
                    delta_z_lists.append(delta_z_list)
                    ft_logits_lists.append(cached_logits)
                    token_ids_lists.append(token_ids)
                
                # Use batch computation function (optimization: transfer all data to GPU at once)
                from evaluate.evaluate_utils import compute_choices_logprobs_from_cache
                mc1_logprobs = compute_choices_logprobs_from_cache(
                    delta_z_lists=delta_z_lists,
                    ft_logits_lists=ft_logits_lists,
                    token_ids_lists=token_ids_lists,
                    lambda_scale=lambda_scale,
                    device=self.device
                )
                mc1_logprobs = np.array(mc1_logprobs)
                correct_answer = mc1_choices[mc1_correct_idx]
                incorrect_answers = [mc1_choices[i] for i in range(len(mc1_choices)) if i != mc1_correct_idx]
                
                mc1_score = evaluator.compute_mc1(
                    mc1_logprobs,
                    correct_answer,
                    [correct_answer],
                    incorrect_answers,
                    mc1_choices
                )
                mc1_scores.append(mc1_score)
            
            # Compute MC2/MC3 (using cached logits, batch optimized)
            if mc2_choices and mc2_correct_choices:
                mc2_choices_data = sample.get("mc2_choices_data", [])
                
                # Batch get delta_z and logits for all choices
                delta_z_lists = []
                ft_logits_lists = []
                token_ids_lists = []
                
                for choice_data in mc2_choices_data:
                    cached_logits = choice_data["logits"]
                    token_ids = choice_data["token_ids"]
                    choice = choice_data["choice"]
                    
                    delta_z_list = self._get_choice_delta_z(
                        sample_delta_store,
                        sample_key,
                        "mc2",
                        choice
                    )
                    
                    # If no delta_z, use mean
                    if delta_z_list is None or len(delta_z_list) == 0:
                        delta_z_list = [delta_z_mean] * len(token_ids)
                    
                    delta_z_lists.append(delta_z_list)
                    ft_logits_lists.append(cached_logits)
                    token_ids_lists.append(token_ids)
                
                # Use batch computation function (optimization: transfer all data to GPU at once)
                from evaluate.evaluate_utils import compute_choices_logprobs_from_cache
                mc2_logprobs = compute_choices_logprobs_from_cache(
                    delta_z_lists=delta_z_lists,
                    ft_logits_lists=ft_logits_lists,
                    token_ids_lists=token_ids_lists,
                    lambda_scale=lambda_scale,
                    device=self.device
                )
                mc2_logprobs = np.array(mc2_logprobs)
                mc2_incorrect_choices = [c for c in mc2_choices if c not in mc2_correct_choices]
                
                mc2_score = evaluator.compute_mc2(
                    mc2_logprobs,
                    mc2_correct_choices,
                    mc2_incorrect_choices,
                    mc2_choices
                )
                mc2_scores.append(mc2_score)
                
                mc3_score = evaluator.compute_mc3(
                    mc2_logprobs,
                    mc2_correct_choices,
                    mc2_incorrect_choices,
                    mc2_choices
                )
                mc3_scores.append(mc3_score)
        
        return {
            "MC1": self._safe_mean(mc1_scores),
            "MC2": self._safe_mean(mc2_scores),
            "MC3": self._safe_mean(mc3_scores)
        }
    
    def _compute_choice_logprob_from_cached_logits(
        self,
        cached_logits: List[torch.Tensor],
        token_ids: List[int],
        lambda_scale: float,
        delta_z_list: List[torch.Tensor],
        delta_z_mean: torch.Tensor
    ) -> float:
        """
        Compute log probability of a choice from cached logits (applying SVD)
        
        Args:
            cached_logits: Logits at each position [vocab_size]
            token_ids: Corresponding token ids
            delta_z_mean: Mean delta_z
            lambda_scale: Calibration strength
        
        Returns:
            Total log probability (sum of all tokens)
        """
        if len(cached_logits) == 0 or len(token_ids) == 0:
            return float('-inf')
        
        seq_len = min(len(token_ids), len(cached_logits))
        if seq_len == 0:
            return float('-inf')
        
        logits_tensor = torch.stack(cached_logits[:seq_len]).to(self.device)
        token_ids_tensor = torch.tensor(token_ids[:seq_len], device=self.device)
        use_mu = abs(lambda_scale) >= 1e-10
        
        if use_mu:
            if delta_z_list is not None and len(delta_z_list) >= seq_len:
                delta_z_tensor = torch.stack(delta_z_list[:seq_len]).to(self.device)
            else:
                delta_z_tensor = delta_z_mean.unsqueeze(0).expand(seq_len, -1).to(self.device)
            adjusted_logits = logits_tensor + lambda_scale * delta_z_tensor
        else:
            adjusted_logits = logits_tensor
        
        log_probs = F.log_softmax(adjusted_logits, dim=-1)
        gathered = log_probs.gather(dim=1, index=token_ids_tensor.unsqueeze(1)).squeeze(1)
        return gathered.sum().item()
    
    def _safe_mean(self, scores: List[float]) -> float:
        """Safely compute mean"""
        if not scores:
            return 0.0
        valid_scores = [s for s in scores if not (np.isnan(s) or np.isinf(s))]
        if not valid_scores:
            return 0.0
        return np.mean(valid_scores) * 100
    
    def _print_comparison_table(self, results: Dict[float, Dict], lambda_values: List[float]):
        """Print comparison table"""
        print("\n" + "=" * 80)
        print("λ Value Comparison Table")
        print("=" * 80)
        
        if self._is_generative_dataset():
            print(f"{'λ value':<10} {'Accuracy':<10}")
            print("-" * 20)
            for lambda_val in lambda_values:
                acc = results[lambda_val].get("accuracy", 0)
                print(f"{lambda_val:<10.4f} {acc:<10.2f}")
        else:
            print(f"{'λ value':<10} {'MC1':<10} {'MC2':<10} {'MC3':<10}")
            print("-" * 40)
            for lambda_val in lambda_values:
                mc1 = results[lambda_val].get("MC1", 0)
                mc2 = results[lambda_val].get("MC2", 0)
                mc3 = results[lambda_val].get("MC3", 0)
                print(f"{lambda_val:<10.4f} {mc1:<10.2f} {mc2:<10.2f} {mc3:<10.2f}")
        
        print("=" * 80)


