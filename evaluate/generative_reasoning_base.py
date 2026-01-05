"""General Generative Reasoning Evaluator"""
import os
import sys
import re
import csv
import json
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm

# Add data directory to path for importing prompt_loader
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)
from prompt_loader import load_prompt_config, format_zero_shot_prompt  # type: ignore

from .evaluate_utils import (
    extract_answer_from_generation,
    normalize_numerical_answer,
    compare_numerical_answers,
    create_question_stopping_criteria,
    get_dataset_stop_string
)


class GenerativeReasoningEvaluator:
    """General evaluator for generative reasoning tasks"""
    
    # Dataset field mapping configuration
    # Define which fields each dataset should use to get question, full answer, and final answer
    DATASET_FIELD_MAPPING = {
        "gsm8k": {
            "question": ["question"],  # Question field (by priority)
            "full_answer": ["answer"],  # Full answer field (by priority)
            "final_answer": ["final_answer"]  # Final answer field (by priority)
        },
        "math500": {
            "question": ["question"],
            "full_answer": ["solution"],
            "final_answer": ["final_answer"]
        },
        "svamp": {
            "question": ["question"],
            "full_answer": ["solution"],
            "final_answer": ["final_answer"]
        },
        "asdiv": {
            "question": ["question"],
            "full_answer": ["solution"],
            "final_answer": ["final_answer"]
        },
        "strategyqa": {
            "question": ["question"],
            "full_answer": [],  # No full solution
            "final_answer": ["answer"]  # answer is the final answer
        },
        "aqua_rat": {
            "question": ["question"],
            "full_answer": ["answer"],
            "final_answer": ["final_answer"]
        }
    }

    def __init__(
        self,
        tokenizer,
        device: str = "cuda",
        dataset_name: str = "math500",
        final_answer_key: str = "final_answer",
        normalize_fn: Optional[Callable[[str], str]] = None,
        compare_fn: Optional[Callable[[str, str], bool]] = None,
        model_name: str = None
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        self.final_answer_key = final_answer_key
        self.prompt_config = load_prompt_config(dataset_name)
        self.normalize_fn = normalize_fn
        self.compare_fn = compare_fn
        self.stop_string = get_dataset_stop_string(dataset_name)
        self.model_name = self._normalize_model_name(model_name) if model_name else "unknown_model"
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to be suitable for filename"""
        if not model_name:
            return "unknown_model"
        # Replace characters unsuitable for filename
        normalized = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        # Extract last part (usually model identifier)
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
        mapping = self.DATASET_FIELD_MAPPING.get(self.dataset_name, {})
        field_names = mapping.get(field_type, [])
        
        # Try to get field value by priority
        for field_name in field_names:
            if field_name in item and item[field_name]:
                return item[field_name]
        
        return ""
    
    def _default_problem_text(self, item: Dict) -> str:
        # First try using mapping configuration
        question = self._get_field_value(item, "question")
        if question:
            return question
        # Fall back to generic fields (compatibility)
        return item.get("problem") or item.get("question") or ""

    def build_prompt(self, item: Dict) -> str:
        base_item = item.copy()
        if "problem" not in base_item:
            base_item["problem"] = self._default_problem_text(item)
        return format_zero_shot_prompt(base_item, self.prompt_config, self.dataset_name)

    def generate_answer(
        self,
        model,
        prompt: str,
        max_new_tokens: int = 1024,  # Output at most 1024 tokens
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1]
        
        # Create stopping criteria: detect dataset-specific termination marker
        stopping_criteria = create_question_stopping_criteria(
            tokenizer=self.tokenizer,
            dataset_name=self.dataset_name,
            initial_input_length=input_length
        )

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )

        generated_tokens = output_ids[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Also filter out content after stop marker after decoding (double insurance)
        if self.stop_string and self.stop_string in generated_text:
            generated_text = generated_text.split(self.stop_string)[0].strip()
        
        return generated_text

    def extract_final_answer(self, generated_text: str) -> str:
        # First filter out content after stop marker (prevent answer extraction from including repeated questions)
        if self.stop_string and self.stop_string in generated_text:
            generated_text = generated_text.split(self.stop_string)[0].strip()
        
        # For math500, asdiv datasets, need to preserve complete answer (may be LaTeX expression, fraction, etc.)
        if self.dataset_name in {"math500", "asdiv"}:
            # Extract complete content after ####
            if "####" in generated_text:
                parts = generated_text.split("####")
                if len(parts) > 1:
                    answer = parts[1].strip()
                    return self.normalize_answer(answer)
            # If no separator, use last line
            lines = generated_text.strip().splitlines()
            answer = lines[-1].strip() if lines else generated_text.strip()
            return self.normalize_answer(answer)
        
        # For other datasets (gsm8k, svamp, etc.), use original number extraction logic
        answer = extract_answer_from_generation(
            generated_text,
            separator="####",
            extract_last_number=False
        )
        if not answer:
            lines = generated_text.strip().splitlines()
            answer = lines[-1].strip() if lines else generated_text.strip()
        return self.normalize_answer(answer)

    def normalize_answer(self, answer: str) -> str:
        if not answer:
            return ""
        result = answer.replace("####", "").strip()
        result = re.sub(r"^(answer|final answer|result)\s*[:：]\s*", "", result, flags=re.IGNORECASE)
        
        # For math500, asdiv, do minimal cleanup, preserve complete format
        if self.dataset_name in {"math500", "asdiv"}:
            # Only remove leading/trailing periods and extra spaces, preserve internal format
            result = result.strip(" .")
        else:
            # For other datasets, normalize whitespace
            result = re.sub(r"\s+", " ", result).strip(" .")
        
        if self.normalize_fn:
            result = self.normalize_fn(result)
        return result

    @staticmethod
    def _looks_numeric(text: str) -> bool:
        """Check if text looks like a numeric value"""
        return any(ch.isdigit() for ch in text)

    def check_correctness(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if predicted answer is correct
        Dynamically select comparison method based on dataset type:
        - math500, asdiv: Exact match (case-insensitive)
        - gsm8k, svamp: Numerical match (if looks like numeric)
        - Other: String match (case-insensitive)
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            is_correct: Whether correct
        """
        if self.compare_fn:
            return self.compare_fn(prediction, ground_truth)
        
        # math500, asdiv datasets: Exact match (case-insensitive)
        if self.dataset_name in {"math500", "asdiv"}:
            return prediction.lower().strip() == ground_truth.lower().strip()
        
        # Other datasets (gsm8k, svamp, etc.): If looks like numeric, use numerical match
        if self._looks_numeric(prediction) or self._looks_numeric(ground_truth):
            if compare_numerical_answers(prediction, ground_truth):
                return True
        
        # Default: String match (case-insensitive)
        return prediction.lower().strip() == ground_truth.lower().strip()

    def evaluate_dataset(
        self,
        model,
        data: List[Dict],
        max_samples: int = None,
        max_new_tokens: int = 1024,  # Output at most 1024 tokens
        temperature: float = 0.7,
        do_sample: bool = False,
        save_outputs: bool = False,
        output_dir: str = "results/case",
        eval_mode: str = "unknown",
        self_consistency: bool = False,
        run_id: int = None
    ) -> Dict[str, float]:
        model.eval()
        eval_data = data[:max_samples] if max_samples else data

        if max_samples:
            print(f"  ⚡ Quick evaluation mode: using first {len(eval_data)}/{len(data)} samples")

        correct = 0
        total = 0
        
        # Prepare data for saving outputs
        output_rows = []  # CSV format (original logic)
        self_consistency_results = []  # JSON format (self-consistency mode)

        for idx, item in enumerate(tqdm(eval_data, desc=f"Evaluating {self.dataset_name}")):
            prompt = item.get("prompt") or self.build_prompt(item)
            generated_text = self.generate_answer(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )

            predicted = self.extract_final_answer(generated_text)
            ground_truth_raw = item.get(self.final_answer_key, "")
            ground_truth = self.normalize_answer(ground_truth_raw)
            
            # Unified field access (using mapping configuration)
            question = self._get_field_value(item, "question")
            if not question:
                # Fall back to generic fields (compatibility)
                question = item.get("problem") or item.get("question", "")
            
            full_answer = self._get_field_value(item, "full_answer")
            final_answer = self._get_field_value(item, "final_answer")
            
            # If final answer not found in mapping config, use ground_truth_raw as fallback
            if not final_answer:
                final_answer = ground_truth_raw

            is_correct = self.check_correctness(predicted, ground_truth)
            if is_correct:
                correct += 1
            total += 1
            
            # If saving outputs needed, collect data (CSV format)
            if save_outputs and not self_consistency:
                output_rows.append({
                    "question": question,
                    "full_answer": full_answer,
                    "final_answer": final_answer,
                    "model_output": generated_text,
                    "extracted_answer": predicted
                })
            
            # Self-consistency mode: save simplified JSON format (only final_answer and ground_truth)
            if self_consistency:
                self_consistency_results.append({
                    "sample_id": idx,
                    "predicted_answer": predicted,
                    "ground_truth": ground_truth,
                    "is_correct": is_correct
                })

        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        # If saving outputs needed, write to CSV file (original logic, non self-consistency mode)
        if save_outputs and output_rows and not self_consistency:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{self.dataset_name}_{eval_mode}_{self.model_name}.csv"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["question", "full_answer", "final_answer", "model_output", "extracted_answer"])
                writer.writeheader()
                writer.writerows(output_rows)
            
            print(f"\n✓ Saved model outputs to: {filepath}")
            print(f"  Saved {len(output_rows)} records")
        
        # Self-consistency mode: save JSON format (only final_answer and ground_truth)
        if self_consistency and self_consistency_results:
            os.makedirs(output_dir, exist_ok=True)
            # Filename: include run_id (if provided)
            if run_id is not None:
                filename = f"{self.dataset_name}_{eval_mode}_run_{run_id}_{self.model_name}.json"
            else:
                filename = f"{self.dataset_name}_{eval_mode}_{self.model_name}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self_consistency_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Saved Self-Consistency results to: {filepath}")
            print(f"  Saved {len(self_consistency_results)} records")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }


def evaluate_with_generation(
    model,
    tokenizer,
    data: List[Dict],
    dataset_name: str,
    final_answer_key: str = "final_answer",
    device: str = "cuda",
    max_samples: int = None,
    max_new_tokens: int = 1024,  # Output at most 1024 tokens
    temperature: float = 0.7,
    do_sample: bool = False,
    normalize_fn: Optional[Callable[[str], str]] = None,
    compare_fn: Optional[Callable[[str], str]] = None
) -> Dict[str, float]:
    evaluator = GenerativeReasoningEvaluator(
        tokenizer=tokenizer,
        device=device,
        dataset_name=dataset_name,
        final_answer_key=final_answer_key,
        normalize_fn=normalize_fn,
        compare_fn=compare_fn
    )
    return evaluator.evaluate_dataset(
        model,
        data,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample
    )

