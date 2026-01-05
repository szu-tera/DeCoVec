"""
QA Format Evaluation System
Adapts to TruthfulQA QA format data
Implements MC1/MC2/MC3 metrics and open-ended generation evaluation
"""
import torch
import torch.nn.functional as F
import sys
import os
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import json

# Add data directory to path for importing prompt_loader
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)
from prompt_loader import load_prompt_config  # type: ignore

# Import general utility functions
from .evaluate_utils import (
    compute_answer_logprob as compute_answer_logprob_util,
    compute_choice_delta_and_logits,
    compute_choices_logprobs_from_cache,
    safe_mean,
    build_choices_logits_cache
)


class QAEvaluator:
    """QA Format Evaluator"""
    
    def __init__(self, tokenizer, device="cuda", dataset_name: str = "truthfulqa"):
        """
        Args:
            tokenizer: Tokenizer
            device: Compute device
            dataset_name: Dataset name (for loading prompt template)
        """
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_name = dataset_name
        # Load prompt configuration
        self.prompt_config = load_prompt_config(dataset_name)
    
    def parse_answers(self, answer_string: str) -> List[str]:
        """
        Parse answer string (semicolon-separated)
        
        Args:
            answer_string: Semicolon-separated answer string
        
        Returns:
            List of answers
        """
        if not answer_string:
            return []
        
        # Split by semicolon and strip whitespace
        answers = [a.strip() for a in answer_string.split(';')]
        # Filter empty strings
        answers = [a for a in answers if a]
        
        return answers
    
    def compute_answer_logprob(
        self,
        model,
        prompt: str,
        answer: str
    ) -> float:
        """
        Compute log probability of an answer
        
        Args:
            model: Language model
            prompt: Few-shot prompt (ending with 'A:')
            answer: Answer text
        
        Returns:
            logprob: Log probability of answer (average of all tokens)
        """
        # Use general utility function
        return compute_answer_logprob_util(model, self.tokenizer, prompt, answer, self.device)
    
    def compute_choice_logprobs(
        self,
        model,
        prompt: str,
        choices: List[str]
    ) -> np.ndarray:
        """
        Compute log probability for each choice
        
        Args:
            model: Language model
            prompt: Few-shot prompt
            choices: List of choices
        
        Returns:
            logprobs: [num_choices] Log probability for each choice
        """
        logprobs = []
        
        for choice in choices:
            logprob = self.compute_answer_logprob(model, prompt, choice)
            logprobs.append(logprob)
        
        return np.array(logprobs)
    
    def compute_choice_logprobs_batch(
        self,
        model,
        prompt: str,
        choices: List[str],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Batch compute log probability for each choice (optimized version using lm-eval standard tokenization)
        
        Args:
            model: Language model
            prompt: Few-shot prompt
            choices: List of choices
            batch_size: Batch size (note: due to varying lengths, actually processed one by one)
        
        Returns:
            logprobs: [num_choices] Log probability for each choice
        """
        if len(choices) == 0:
            return np.array([])
        
        logprobs = []
        
        # Use standard lm-eval tokenization method
        # Note: Due to varying continuation lengths, cannot truly batch, but keep interface consistent
        for choice in choices:
            logprob = self.compute_answer_logprob(model, prompt, choice)
            logprobs.append(logprob)
        
        return np.array(logprobs)
    
    def compute_mc1(
        self,
        logprobs: np.ndarray,
        best_answer: str,
        correct_answers: List[str],
        incorrect_answers: List[str],
        all_choices: List[str]
    ) -> float:
        """
        MC1: 1vFalse -- best correct answer vs all false answers
        Check if best answer score is greater than max score of all incorrect answers
        
        Args:
            logprobs: [num_choices] Log probabilities
            best_answer: Best correct answer
            correct_answers: List of correct answers
            incorrect_answers: List of incorrect answers
            all_choices: List of all choices
        
        Returns:
            accuracy: 0 or 1
        """
        if len(correct_answers) == 0 or len(incorrect_answers) == 0:
            return 0.0
        
        # Check if best answer is in choices
        if best_answer not in all_choices:
            return 0.0
        
        # Find index of best answer
        best_idx = all_choices.index(best_answer)
        
        # Find scores of all incorrect answers
        incorrect_scores = []
        for ans in incorrect_answers:
            if ans in all_choices:
                idx = all_choices.index(ans)
                incorrect_scores.append(logprobs[idx])
        
        if len(incorrect_scores) == 0:
            return 0.0
        
        # Compare best answer score with max score of all incorrect answers
        best_score = logprobs[best_idx]
        max_incorrect_score = max(incorrect_scores)
        
        return 1.0 if best_score > max_incorrect_score else 0.0
    
    def compute_mc2(
        self,
        logprobs: np.ndarray,
        correct_answers: List[str],
        incorrect_answers: List[str],
        all_choices: List[str]
    ) -> float:
        """
        MC2: normalized probability mass for correct answers
        Compute normalized probability mass (probability sum) for correct answers
        
        Args:
            logprobs: [num_choices] Log probabilities
            correct_answers: List of correct answers
            incorrect_answers: List of incorrect answers
            all_choices: List of all choices
        
        Returns:
            normalized_prob: Normalized probability sum for correct answers
        """
        if len(correct_answers) == 0 or len(incorrect_answers) == 0:
            return 0.0
        
        # Find scores of all correct answers
        correct_scores = []
        for ans in correct_answers:
            if ans in all_choices:
                idx = all_choices.index(ans)
                correct_scores.append(logprobs[idx])
        
        # Find scores of all incorrect answers
        incorrect_scores = []
        for ans in incorrect_answers:
            if ans in all_choices:
                idx = all_choices.index(ans)
                incorrect_scores.append(logprobs[idx])
        
        if len(correct_scores) == 0 or len(incorrect_scores) == 0:
            return 0.0
        
        # Convert to probabilities
        probs_true = np.exp(correct_scores)
        probs_false = np.exp(incorrect_scores)
        
        # Normalize: correct answer probability / (sum of all correct answer probabilities + sum of all incorrect answer probabilities)
        total_prob = sum(probs_true) + sum(probs_false)
        if total_prob == 0:
            return 0.0
        
        probs_true_normalized = probs_true / total_prob
        
        # Return normalized sum of correct answer probabilities
        return float(sum(probs_true_normalized))
    
    def compute_mc3(
        self,
        logprobs: np.ndarray,
        correct_answers: List[str],
        incorrect_answers: List[str],
        all_choices: List[str]
    ) -> float:
        """
        MC3: 1vFalse -- each correct answer vs all false answers
        Count how many correct answers have scores greater than max score of all incorrect answers, then average
        
        Args:
            logprobs: [num_choices] Log probabilities
            correct_answers: List of correct answers
            incorrect_answers: List of incorrect answers
            all_choices: List of all choices
        
        Returns:
            ratio: Ratio of correct answers exceeding all incorrect answers
        """
        if len(correct_answers) == 0 or len(incorrect_answers) == 0:
            return 0.0
        
        # Find scores of all correct answers
        correct_scores = []
        for ans in correct_answers:
            if ans in all_choices:
                idx = all_choices.index(ans)
                correct_scores.append(logprobs[idx])
        
        # Find scores of all incorrect answers
        incorrect_scores = []
        for ans in incorrect_answers:
            if ans in all_choices:
                idx = all_choices.index(ans)
                incorrect_scores.append(logprobs[idx])
        
        if len(correct_scores) == 0 or len(incorrect_scores) == 0:
            return 0.0
        
        # Count how many correct answers have scores greater than max score of all incorrect answers
        max_incorrect_score = max(incorrect_scores)
        num_correct_above_max = sum(1 for score in correct_scores if score > max_incorrect_score)
        
        # Return ratio
        return float(num_correct_above_max) / len(correct_scores)
    
    def evaluate_dataset(
        self,
        model,
        data: List[Dict],
        compute_mc1: bool = True,
        compute_mc2: bool = True,
        compute_mc3: bool = True,
        batch_size: int = 1,
        max_samples: int = None,
        use_batch_compute: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate on entire dataset
        
        Args:
            model: Language model
            data: Dataset (QA format)
            compute_mc1/2/3: Whether to compute corresponding metrics
            batch_size: Batch size (for choice computation)
            max_samples: Maximum evaluation samples, None means all
            use_batch_compute: Whether to use batch computation (faster but higher memory usage)
        
        Returns:
            metrics: Evaluation metrics dictionary
        """
        mc1_scores = []
        mc2_scores = []
        mc3_scores = []
        
        model.eval()
        
        # Sample data
        eval_data = data[:max_samples] if max_samples else data
        
        if max_samples:
            print(f"  ⚡ Quick evaluation mode: using first {len(eval_data)}/{len(data)} samples")
        
        for item in tqdm(eval_data, desc="Evaluating QA (MC)"):
            prompt = item["prompt"]
            best_answer = item.get("best_answer", "")
            
            # MC1 evaluation: use mc1_choices (contains 1 correct answer and multiple incorrect answers)
            if compute_mc1:
                mc1_choices = item.get("mc1_choices", [])
                mc1_correct_idx = item.get("mc1_correct_idx", None)
                
                if len(mc1_choices) > 0 and mc1_correct_idx is not None:
                    # Compute log probabilities for all MC1 choices
                    if use_batch_compute and batch_size > 1:
                        mc1_logprobs = self.compute_choice_logprobs_batch(model, prompt, mc1_choices, batch_size)
                    else:
                        mc1_logprobs = self.compute_choice_logprobs(model, prompt, mc1_choices)
                    
                    # Separate correct and incorrect answers
                    correct_answer = mc1_choices[mc1_correct_idx]
                    incorrect_answers = [mc1_choices[i] for i in range(len(mc1_choices)) if i != mc1_correct_idx]
                    
                    # MC1: best answer vs all false answers
                    mc1_score = self.compute_mc1(
                        mc1_logprobs, 
                        correct_answer,  # Use correct answer from mc1 as best answer
                        [correct_answer],
                        incorrect_answers,
                        mc1_choices
                    )
                    mc1_scores.append(mc1_score)
            
            # MC2/MC3 evaluation: use mc2_choices (contains multiple correct and multiple incorrect answers)
            if compute_mc2 or compute_mc3:
                mc2_choices = item.get("mc2_choices", [])
                mc2_correct_choices = item.get("mc2_correct_choices", [])
                
                if len(mc2_choices) > 0 and len(mc2_correct_choices) > 0:
                    # Compute log probabilities for all MC2 choices
                    if use_batch_compute and batch_size > 1:
                        mc2_logprobs = self.compute_choice_logprobs_batch(model, prompt, mc2_choices, batch_size)
                    else:
                        mc2_logprobs = self.compute_choice_logprobs(model, prompt, mc2_choices)
                    
                    # Get list of incorrect answers
                    mc2_incorrect_choices = [c for c in mc2_choices if c not in mc2_correct_choices]
                    
                    # MC2: normalized probability mass for correct answers
                    if compute_mc2:
                        mc2_score = self.compute_mc2(
                            mc2_logprobs, 
                            mc2_correct_choices, 
                            mc2_incorrect_choices,
                            mc2_choices
                        )
                        mc2_scores.append(mc2_score)
                    
                    # MC3: each correct answer vs all false answers
                    if compute_mc3:
                        mc3_score = self.compute_mc3(
                            mc2_logprobs, 
                            mc2_correct_choices, 
                            mc2_incorrect_choices,
                            mc2_choices
                        )
                        mc3_scores.append(mc3_score)
        
        # Aggregate results (use general utility function for mean calculation)
        metrics = {}
        if compute_mc1:
            metrics["MC1"] = safe_mean(mc1_scores)
        if compute_mc2:
            metrics["MC2"] = safe_mean(mc2_scores)
        if compute_mc3:
            metrics["MC3"] = safe_mean(mc3_scores)
        
        return metrics


def build_logits_cache(
    base_model,
    finetuned_model,
    steering_computer,
    tokenizer,
    data: List[Dict],
    device="cuda",
    method: str = "standard",
    max_samples: int = None
) -> Dict:
    """
    Build logits cache (cache delta_z and ft_logits)
    For fast testing of different μ values later
    
    Args:
        base_model: Original model
        finetuned_model: Fine-tuned model
        steering_computer: SteeringVectorComputer instance
        tokenizer: Tokenizer
        data: Evaluation data (QA format)
        device: Device
        method: SVD method
        max_samples: Maximum number of samples
    
    Returns:
        cache_dict: {
            "samples": [
                {
                    "mc1_choices": [...],
                    "mc1_correct_idx": ...,
                    "mc1_delta_z_list": [delta_z tensor for each choice],
                    "mc1_ft_logits_list": [ft_logits tensor for each choice],
                    "mc2_choices": [...],
                    "mc2_correct_choices": [...],
                    "mc2_delta_z_list": [delta_z tensor for each choice],
                    "mc2_ft_logits_list": [ft_logits tensor for each choice],
                }
            ]
        }
    """
    from decovec.decovec_core import SteeringVectorComputer
    
    base_model.eval()
    finetuned_model.eval()
    
    eval_data = data[:max_samples] if max_samples else data
    
    if max_samples:
        print(f"  ⚡ Quick mode: using first {len(eval_data)}/{len(data)} samples")
    
    cache_samples = []
    
    for item in tqdm(eval_data, desc="Building Logits Cache"):
        prompt = item["prompt"]
        sample_cache = {}
        
        # Process MC1 choices
        mc1_choices = item.get("mc1_choices", [])
        mc1_correct_idx = item.get("mc1_correct_idx", None)
        
        if len(mc1_choices) > 0 and mc1_correct_idx is not None:
            sample_cache["mc1_choices"] = mc1_choices
            sample_cache["mc1_correct_idx"] = mc1_correct_idx
            sample_cache["mc1_delta_z_list"] = []
            sample_cache["mc1_ft_logits_list"] = []
            
            sample_cache["mc1_token_ids_list"] = []
            
            # Use general function to build cache for all MC1 choices
            delta_z_lists, ft_logits_lists, token_ids_lists = build_choices_logits_cache(
                mc1_choices, prompt, base_model, finetuned_model, steering_computer,
                tokenizer, device, method
            )
            sample_cache["mc1_delta_z_list"] = delta_z_lists
            sample_cache["mc1_ft_logits_list"] = ft_logits_lists
            sample_cache["mc1_token_ids_list"] = token_ids_lists
        
        # Process MC2 choices
        mc2_choices = item.get("mc2_choices", [])
        mc2_correct_choices = item.get("mc2_correct_choices", [])
        
        if len(mc2_choices) > 0 and len(mc2_correct_choices) > 0:
            sample_cache["mc2_choices"] = mc2_choices
            sample_cache["mc2_correct_choices"] = mc2_correct_choices
            sample_cache["mc2_delta_z_list"] = []
            sample_cache["mc2_ft_logits_list"] = []
            sample_cache["mc2_token_ids_list"] = []
            
            # Use general function to build cache for all MC2 choices
            delta_z_lists, ft_logits_lists, token_ids_lists = build_choices_logits_cache(
                mc2_choices, prompt, base_model, finetuned_model, steering_computer,
                tokenizer, device, method
            )
            sample_cache["mc2_delta_z_list"] = delta_z_lists
            sample_cache["mc2_ft_logits_list"] = ft_logits_lists
            sample_cache["mc2_token_ids_list"] = token_ids_lists
        
        cache_samples.append(sample_cache)
    
    return {"samples": cache_samples, "method": method}


def evaluate_with_logits_cache(
    logits_cache: Dict,
    lambda_scale: float,
    tokenizer,
    device="cuda",
    dataset_name: str = "truthfulqa"
) -> Dict[str, float]:
    """
    Use cached logits for SVD evaluation (supports fast testing of different λ values)
    
    Args:
        logits_cache: Cache returned by build_logits_cache
        lambda_scale: Global calibration strength λ
        tokenizer: Tokenizer
        device: Device
    
    Returns:
        metrics: Evaluation metrics
    """
    evaluator = QAEvaluator(tokenizer, device, dataset_name=dataset_name)
    
    cache_samples = logits_cache["samples"]
    
    mc1_scores = []
    mc2_scores = []
    mc3_scores = []
    
    debug_count = 0
    debug_logprobs_sample = []
    
    for sample_cache in tqdm(cache_samples, desc=f"Evaluating (λ={lambda_scale:.4f})"):
        # MC1 evaluation
        if "mc1_delta_z_list" in sample_cache:
            mc1_choices = sample_cache["mc1_choices"]
            mc1_correct_idx = sample_cache["mc1_correct_idx"]
            
            # Batch compute log probabilities for all MC1 choices (using general utility function)
            mc1_logprobs = compute_choices_logprobs_from_cache(
                sample_cache["mc1_delta_z_list"],
                sample_cache["mc1_ft_logits_list"],
                sample_cache["mc1_token_ids_list"],
                lambda_scale,
                device
            )
            mc1_logprobs = np.array(mc1_logprobs)
            
            # Debug: record logprobs for first few samples
            if debug_count < 3:
                debug_logprobs_sample.append({
                    'mc1_logprobs': mc1_logprobs.tolist(),
                    'mc1_correct_idx': mc1_correct_idx
                })
                debug_count += 1
            
            # Separate correct and incorrect answers
            correct_answer = mc1_choices[mc1_correct_idx]
            incorrect_answers = [mc1_choices[i] for i in range(len(mc1_choices)) if i != mc1_correct_idx]
            
            # MC1 scoring
            mc1_score = evaluator.compute_mc1(
                mc1_logprobs,
                correct_answer,
                [correct_answer],
                incorrect_answers,
                mc1_choices
            )
            mc1_scores.append(mc1_score)
        
        # MC2/MC3 evaluation
        if "mc2_delta_z_list" in sample_cache:
            mc2_choices = sample_cache["mc2_choices"]
            mc2_correct_choices = sample_cache["mc2_correct_choices"]
            
            # Batch compute log probabilities for all MC2 choices (using general utility function)
            mc2_logprobs = compute_choices_logprobs_from_cache(
                sample_cache["mc2_delta_z_list"],
                sample_cache["mc2_ft_logits_list"],
                sample_cache["mc2_token_ids_list"],
                lambda_scale,
                device
            )
            mc2_logprobs = np.array(mc2_logprobs)
            
            # Get list of incorrect answers
            mc2_incorrect_choices = [c for c in mc2_choices if c not in mc2_correct_choices]
            
            # MC2 scoring
            mc2_score = evaluator.compute_mc2(
                mc2_logprobs,
                mc2_correct_choices,
                mc2_incorrect_choices,
                mc2_choices
            )
            mc2_scores.append(mc2_score)
            
            # MC3 scoring
            mc3_score = evaluator.compute_mc3(
                mc2_logprobs,
                mc2_correct_choices,
                mc2_incorrect_choices,
                mc2_choices
            )
            mc3_scores.append(mc3_score)
    
    # Compute metrics (using general utility function)
    metrics = {
        "MC1": safe_mean(mc1_scores),
        "MC2": safe_mean(mc2_scores),
        "MC3": safe_mean(mc3_scores)
    }
    
    return metrics


def evaluate_with_svd(
    base_model,
    finetuned_model,
    svd_decoder,
    tokenizer,
    data: List[Dict],
    device="cuda",
    batch_size: int = 1,
    max_samples: int = None,
    dataset_name: str = "truthfulqa"
) -> Dict[str, float]:
    """
    Evaluate using SVD decoding (QA format)
    
    Args:
        base_model: Original model
        finetuned_model: Fine-tuned model
        svd_decoder: SVD decoder
        tokenizer: Tokenizer
        data: Evaluation data (QA format)
        device: Device
    
    Returns:
        metrics: Evaluation metrics
    """
    evaluator = QAEvaluator(tokenizer, device, dataset_name=dataset_name)
    
    def compute_svd_answer_logprob(prompt: str, answer: str) -> float:
        """Compute answer log probability using SVD adjustment"""
        # Use unified tokenization method (lm-eval's _encode_pair)
        from evaluate_utils import tokenize_prompt_and_continuation
        context_enc, continuation_enc = tokenize_prompt_and_continuation(
            tokenizer, prompt, answer
        )
        
        if len(continuation_enc) == 0:
            return float('-inf')
        
        # Construct input: concatenate and remove last token
        inp = (context_enc + continuation_enc)[:-1]
        
        # Get logits from both models
        with torch.no_grad():
            input_ids = torch.tensor([inp]).to(device)
            
            base_outputs = base_model(input_ids)
            base_logits = base_outputs.logits[0]
            
            ft_outputs = finetuned_model(input_ids)
            ft_logits = ft_outputs.logits[0]
        
        # Select continuation part logits
        inplen = len(inp)
        contlen = len(continuation_enc)
        cont_base_logits = base_logits[inplen - contlen : inplen]
        cont_ft_logits = ft_logits[inplen - contlen : inplen]
        
        # Compute answer log probability (using SVD adjustment)
        answer_logprob = 0.0
        for i, token_id in enumerate(continuation_enc):
            # Apply SVD adjustment
            adjusted_logits = svd_decoder.adjust_logits(
                cont_base_logits[i].unsqueeze(0),
                cont_ft_logits[i].unsqueeze(0)
            )
            adjusted_probs = F.softmax(adjusted_logits[0], dim=-1)
            token_logprob = torch.log(adjusted_probs[token_id] + 1e-12)
            answer_logprob += token_logprob.item()
        
        # Return sum (reference lm-evaluation-harness, no normalization)
        return answer_logprob
    
    def compute_svd_logprobs(prompt: str, choices: List[str]) -> np.ndarray:
        """Compute choice log probabilities using SVD adjustment"""
        logprobs = []
        for choice in choices:
            logprob = compute_svd_answer_logprob(prompt, choice)
            logprobs.append(logprob)
        return np.array(logprobs)
    
    mc1_scores = []
    mc2_scores = []
    mc3_scores = []
    
    base_model.eval()
    finetuned_model.eval()
    
    # Sample data
    eval_data = data[:max_samples] if max_samples else data
    
    if max_samples:
        print(f"  ⚡ Quick evaluation mode: using first {len(eval_data)}/{len(data)} samples")
    
    for item in tqdm(eval_data, desc="Evaluating QA (SVD)"):
        prompt = item["prompt"]
        best_answer = item.get("best_answer", "")
        
        # MC1 evaluation: use mc1_choices (contains 1 correct answer and multiple incorrect answers)
        mc1_choices = item.get("mc1_choices", [])
        mc1_correct_idx = item.get("mc1_correct_idx", None)
        
        if len(mc1_choices) > 0 and mc1_correct_idx is not None:
            # Compute SVD-adjusted MC1 log probabilities
            mc1_logprobs = compute_svd_logprobs(prompt, mc1_choices)
            
            # Separate correct and incorrect answers
            correct_answer = mc1_choices[mc1_correct_idx]
            incorrect_answers = [mc1_choices[i] for i in range(len(mc1_choices)) if i != mc1_correct_idx]
            
            # MC1: best answer vs all false answers
            mc1_score = evaluator.compute_mc1(
                mc1_logprobs, 
                correct_answer,  # Use correct answer from mc1 as best answer
                [correct_answer],
                incorrect_answers,
                mc1_choices
            )
            mc1_scores.append(mc1_score)
        
        # MC2/MC3 evaluation: use mc2_choices (contains multiple correct and multiple incorrect answers)
        mc2_choices = item.get("mc2_choices", [])
        mc2_correct_choices = item.get("mc2_correct_choices", [])
        
        if len(mc2_choices) > 0 and len(mc2_correct_choices) > 0:
            # Compute SVD-adjusted MC2 log probabilities
            mc2_logprobs = compute_svd_logprobs(prompt, mc2_choices)
            
            # Get list of incorrect answers
            mc2_incorrect_choices = [c for c in mc2_choices if c not in mc2_correct_choices]
            
            # MC2: normalized probability mass for correct answers
            mc2_score = evaluator.compute_mc2(
                mc2_logprobs, 
                mc2_correct_choices, 
                mc2_incorrect_choices,
                mc2_choices
            )
            mc2_scores.append(mc2_score)
            
            # MC3: each correct answer vs all false answers
            mc3_score = evaluator.compute_mc3(
                mc2_logprobs, 
                mc2_correct_choices, 
                mc2_incorrect_choices,
                mc2_choices
            )
            mc3_scores.append(mc3_score)
    
    # Compute metrics (using general utility function)
    metrics = {
        "MC1": safe_mean(mc1_scores),
        "MC2": safe_mean(mc2_scores),
        "MC3": safe_mean(mc3_scores)
    }
    
    return metrics


def evaluate_generation(
    model,
    tokenizer,
    data: List[Dict],
    device="cuda",
    max_new_tokens: int = 1024,  # Output at most 1024 tokens
    temperature: float = 1.0,
    dataset_name: str = "gsm8k"
) -> List[Dict]:
    """
    Evaluate open-ended generation
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        data: Evaluation data
        device: Device
        max_new_tokens: Maximum generation tokens
        temperature: Generation temperature
    
    Returns:
        results: List of generation results, containing question, prompt, generated_answer, best_answer
    """
    model.eval()
    results = []
    
    for item in tqdm(data, desc="Generating answers"):
        prompt = item["prompt"]
        question = item["question"]
        best_answer = item["best_answer"]
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_length = input_ids.shape[1]
        
        # Create stopping criteria: determine truncation marker based on dataset
        from evaluate.evaluate_utils import create_question_stopping_criteria, get_dataset_stop_string
        stop_marker = get_dataset_stop_string(dataset_name)
        stopping_criteria = create_question_stopping_criteria(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            initial_input_length=input_length
        )
        
        # Generate answer
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,  # Use greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
        
        # Decode generated answer
        generated_tokens = output_ids[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Also filter out content after stop marker after decoding (double insurance)
        if stop_marker and stop_marker in generated_text:
            generated_text = generated_text.split(stop_marker)[0].strip()
        
        # Extract generated answer part (remove prompt)
        generated_answer = generated_text
        
        results.append({
            "question": question,
            "prompt": prompt,
            "generated_answer": generated_answer,
            "best_answer": best_answer,
            "correct_answers": item.get("correct_answers", ""),
            "incorrect_answers": item.get("incorrect_answers", "")
        })
    
    return results


if __name__ == "__main__":
    # Test code
    print("QA evaluation module loaded successfully")

