"""
General Evaluation Utility Functions Library
Contains shared functions across evaluation modules
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
import re
from transformers import StoppingCriteria, StoppingCriteriaList


__all__ = [
    'compute_answer_logprob',
    'compute_choice_delta_and_logits',
    'compute_accuracy',
    'safe_mean',
    'extract_answer_from_generation',
    'normalize_numerical_answer',
    'extract_number_from_string',
    'compare_numerical_answers',
    'compute_choice_logprob_from_cache',
    'compute_choices_logprobs_from_cache',
    'build_choices_logits_cache',
    'predict_from_logprobs',
    'get_dataset_stop_string',
    'QuestionStoppingCriteria',
    'create_question_stopping_criteria'
]

DEFAULT_STOP_STRING = "Question:"
DATASET_STOP_STRINGS = {
    "truthfulqa": "Q:",
    "boolq": "question:",
    "gsm8k": "Question:",
    "commonsense_qa": "Question:",
    "strategyqa": "Facts:",
    "math500": "Question:",
    "svamp": "Question:",
    "asdiv": "Question:",
    "aqua_rat": "Question:",
    "news_factor": "Article:",
}


def get_dataset_stop_string(dataset_name: Optional[str]) -> str:
    if dataset_name:
        return DATASET_STOP_STRINGS.get(dataset_name.lower(), DEFAULT_STOP_STRING)
    return DEFAULT_STOP_STRING


def tokenize_prompt_and_continuation(
    tokenizer,
    prompt: str,
    continuation: str
) -> Tuple[List[int], List[int]]:
    """
    Standard prompt + continuation tokenization (lm-evaluation-harness method)
    
    Reference lm-eval's _encode_pair implementation, key steps:
    1. Move trailing whitespace from context to continuation start (avoid tokenization changes due to whitespace)
    2. Tokenize complete text: whole_enc = tokenize(context + continuation)
    3. Tokenize context: context_enc = tokenize(context)
    4. Get true continuation tokens via slicing: continuation_enc = whole_enc[len(context_enc):]
    
    This ensures continuation tokenization is consistent with complete text, avoiding whitespace issues.
    
    Args:
        tokenizer: Tokenizer
        prompt: Prompt (context)
        continuation: Answer text (continuation)
    
    Returns:
        context_enc: Token IDs for context
        continuation_enc: True token IDs for continuation (sliced from complete text)
    """
    # Key step 1: Move trailing whitespace from context to continuation start
    # This avoids inconsistency between tokenize(context) + tokenize(continuation) and tokenize(context + continuation)
    n_spaces = len(prompt) - len(prompt.rstrip())
    if n_spaces > 0:
        continuation = prompt[-n_spaces:] + continuation
        prompt = prompt[:-n_spaces]
    
    # Key step 2: First tokenize complete text, then tokenize context
    whole_enc = tokenizer.encode(prompt + continuation, add_special_tokens=True)
    context_enc = tokenizer.encode(prompt, add_special_tokens=True)
    
    # Key step 3: Get true continuation tokens via slicing
    # This gives continuation_enc as the true tokenization in complete context
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    
    return context_enc, continuation_enc


def compute_answer_logprob(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: str = "cuda"
) -> float:
    """
    Compute log probability of an answer (reference lm-evaluation-harness implementation)
    
    Key logic (reference lm-eval's _loglikelihood_tokens):
    1. Use _encode_pair method to get context_enc and continuation_enc
    2. Model input: inp = (context_enc + continuation_enc)[:-1]
    3. Model outputs logits, each position predicts next token
    4. Select continuation part logits: logits[inplen - contlen : inplen]
    5. Compute log_softmax and extract log probability of target token
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Prompt
        answer: Answer text
        device: Compute device
    
    Returns:
        logprob: Log probability of answer (sum of all tokens)
    """
    # Use standard tokenization method (lm-eval's _encode_pair)
    context_enc, continuation_enc = tokenize_prompt_and_continuation(
        tokenizer, prompt, answer
    )
    
    if len(continuation_enc) == 0:
        return float('-inf')
    
    # Construct input: concatenate and remove last token
    # Reference lm-eval huggingface.py line 1232
    inp = (context_enc + continuation_enc)[:-1]
    
    # Forward pass
    with torch.no_grad():
        input_tensor = torch.tensor([inp], dtype=torch.long).to(device)
        outputs = model(input_tensor)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    # Compute log_softmax (over entire vocabulary)
    # Reference lm-eval huggingface.py line 1297
    log_probs = F.log_softmax(logits, dim=-1)  # [seq_len, vocab_size]
    
    # Select continuation part logits
    # Reference lm-eval huggingface.py line 1317 and _select_cont_toks line 1019
    inplen = len(inp)
    contlen = len(continuation_enc)
    
    # Logits positions corresponding to continuation
    cont_log_probs = log_probs[inplen - contlen : inplen]  # [contlen, vocab_size]
    
    # Extract log probability for each continuation token
    # Reference lm-eval huggingface.py line 1346
    answer_logprob = 0.0
    for i, token_id in enumerate(continuation_enc):
        token_log_prob = cont_log_probs[i, token_id]
        answer_logprob += token_log_prob.item()
    
    # Return total log probability (lm-eval line 1351 returns sum)
    return answer_logprob


def compute_choice_delta_and_logits(
    prompt: str,
    choice: str,
    base_model,
    finetuned_model,
    steering_computer,
    tokenizer,
    device: str,
    method: str
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    Compute delta_z and ft_logits over entire vocabulary for each token position of a single choice
    (reference lm-evaluation-harness implementation)
    
    This is the correct SVD decoding preprocessing:
    - For each position, extract logits over entire vocabulary (shape: [vocab_size])
    - Compute delta_z over entire vocabulary dimension
    - Save complete ft_logits and delta_z for subsequent adjustment
    
    Args:
        prompt: Prompt
        choice: Choice text
        base_model: Original model
        finetuned_model: Fine-tuned model
        steering_computer: SteeringVectorComputer instance
        tokenizer: Tokenizer
        device: Device
        method: SVD method
    
    Returns:
        delta_z_tokens: List[delta_z] delta_z over entire vocabulary for each position (each shape: [vocab_size])
        ft_logits_tokens: List[ft_logits] ft_logits over entire vocabulary for each position (each shape: [vocab_size])
        answer_token_ids: List[token_id] target token ID for each position
    """
    # Use standard tokenization method (lm-eval's _encode_pair)
    context_enc, continuation_enc = tokenize_prompt_and_continuation(
        tokenizer, prompt, choice
    )
    
    if len(continuation_enc) == 0:
        return [], [], []
    
    # Construct input: concatenate and remove last token
    # Reference lm-eval huggingface.py line 1232
    inp = (context_enc + continuation_enc)[:-1]
    
    # Get logits from both models
    with torch.no_grad():
        input_tensor = torch.tensor([inp], dtype=torch.long).to(device)
        
        base_outputs = base_model(input_tensor)
        base_logits = base_outputs.logits[0]  # shape: [seq_len, vocab_size]
        
        ft_outputs = finetuned_model(input_tensor)
        ft_logits = ft_outputs.logits[0]      # shape: [seq_len, vocab_size]
    
    # Select continuation part logits
    # Reference lm-eval: logits[inplen - contlen : inplen]
    inplen = len(inp)
    contlen = len(continuation_enc)
    
    # Logits positions corresponding to continuation
    cont_base_logits = base_logits[inplen - contlen : inplen]  # [contlen, vocab_size]
    cont_ft_logits = ft_logits[inplen - contlen : inplen]      # [contlen, vocab_size]
    
    delta_z_tokens = []
    ft_logits_tokens = []
    answer_token_ids = []
    
    for i, token_id in enumerate(continuation_enc):
        # Compute delta_z over entire vocabulary dimension (input and output are both [vocab_size])
        delta_z = steering_computer.compute_delta_z(
            cont_base_logits[i],  # shape: [vocab_size] - entire vocabulary!
            cont_ft_logits[i],    # shape: [vocab_size] - entire vocabulary!
            method=method
        )
        
        # Save entire vocabulary tensors (each is [vocab_size])
        delta_z_tokens.append(delta_z.cpu())              # [vocab_size]
        ft_logits_tokens.append(cont_ft_logits[i].cpu())  # [vocab_size]
        answer_token_ids.append(token_id)                 # Just record target token ID
    
    return delta_z_tokens, ft_logits_tokens, answer_token_ids


def compute_accuracy(
    predictions: List[str],
    ground_truths: List[str]
) -> float:
    """
    Compute accuracy
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
    
    Returns:
        accuracy: Accuracy (0-100)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Number of predictions ({len(predictions)}) does not match number of ground truths ({len(ground_truths)})")
    
    correct = sum(1 for pred, true in zip(predictions, ground_truths) if pred == true)
    accuracy = correct / len(predictions) * 100 if len(predictions) > 0 else 0.0
    
    return accuracy


def safe_mean(scores: List[float]) -> float:
    """
    Safely compute mean, filtering nan and inf
    
    Args:
        scores: List of scores
    
    Returns:
        mean: Mean value
    """
    if not scores:
        return 0.0
    valid_scores = [s for s in scores if not (np.isnan(s) or np.isinf(s))]
    if not valid_scores:
        return 0.0
    return np.mean(valid_scores) * 100


def extract_answer_from_generation(
    generated_text: str,
    separator: str = "####",
    extract_last_number: bool = True
) -> str:
    """
    Extract answer from generated text
    
    Extract first number after ####, then truncate
    
    Args:
        generated_text: Generated text
        separator: Separator (e.g., "####")
        extract_last_number: Whether to extract last number if separator not found
    
    Returns:
        answer: Extracted answer
    """
    # If separator exists, extract first number after separator
    if separator in generated_text:
        # Get content after first separator
        parts = generated_text.split(separator)
        if len(parts) > 1:
            after_separator = parts[1]
            
            # Find first number after #### (supports negative, decimal, thousands separator)
            # Pattern: optional negative sign + digits + optional comma and decimal point
            match = re.search(r'(-?\$?\s*\d[\d,]*\.?\d*%?)', after_separator)
            if match:
                answer = match.group(1).strip()
                # Clean common math symbols
                answer = answer.replace(",", "").replace("$", "").replace("%", "").strip()
                return answer
            
            # If no number found, try to extract first word (might be text answer)
            first_word = after_separator.split()[0] if after_separator.split() else ""
            if first_word:
                return first_word.replace(",", "").replace("$", "").replace("%", "").strip()
    
    # If no separator and number extraction needed
    if extract_last_number:
        # Try to extract last number
        numbers = re.findall(r'-?\d+\.?\d*', generated_text)
        if numbers:
            return numbers[-1]
    
    return ""


def normalize_numerical_answer(answer: str) -> str:
    """
    Normalize numerical answer
    
    Args:
        answer: Original answer string
    
    Returns:
        normalized: Normalized answer
    """
    if not answer:
        return ""
    
    # Remove common math symbols and format characters
    answer = answer.replace(",", "")  # Thousands separator
    answer = answer.replace("$", "")  # Dollar sign
    answer = answer.replace("%", "")  # Percent sign
    
    # Strip leading/trailing whitespace
    answer = answer.strip()
    
    # Check if contains digits
    if not answer or not any(c.isdigit() for c in answer):
        return answer
    
    # First try direct conversion to number
    try:
        num = float(answer)
        # If integer, return integer format
        if num.is_integer():
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        # If direct conversion fails, try extracting number from string
        extracted = extract_number_from_string(answer)
        if extracted != answer:
            # If extracted different string, recursively process extracted number
            return normalize_numerical_answer(extracted)
        else:
            # If cannot extract number, return original string
            return answer


def extract_number_from_string(text: str) -> str:
    """
    Extract number from string
    
    Args:
        text: Input text
    
    Returns:
        number: Extracted number string, or original text if none
    """
    if not text:
        return text
    
    # Try to find last number (supports negative and decimal)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return text


def compare_numerical_answers(pred: str, truth: str, tolerance: float = 1e-6) -> bool:
    """
    Compare two numerical answers for equality
    Reference GSM8K standard evaluation logic:
    1. Direct string comparison
    2. Extract numbers then compare
    3. Float comparison
    
    Args:
        pred: Predicted answer
        truth: Ground truth answer
        tolerance: Float comparison tolerance
    
    Returns:
        is_correct: Whether correct
    """
    # Normalize answers
    pred_norm = normalize_numerical_answer(pred)
    truth_norm = normalize_numerical_answer(truth)
    
    # 1. First try direct string comparison
    if pred_norm == truth_norm:
        return True
    
    # 2. Try comparison after extracting numbers
    pred_extracted = extract_number_from_string(pred_norm)
    if pred_extracted == truth_norm:
        return True
    
    # 3. Numerical comparison (supports floats)
    # Check if both contain digits
    if not (any(c.isdigit() for c in pred_extracted) and any(c.isdigit() for c in truth_norm)):
        return False
    
    pred_num = float(pred_extracted)
    truth_num = float(truth_norm)
    return abs(pred_num - truth_num) < tolerance


def compute_choice_logprob_from_cache(
    delta_z_list: List[torch.Tensor],
    ft_logits_list: List[torch.Tensor],
    token_ids_list: List[int],
    lambda_scale: float,
    device: str = "cuda"
) -> float:
    """
    Compute log probability for a single choice from cached data (with SVD adjustment)
    (reference lm-evaluation-harness implementation)
    
    Correct SVD decoding flow:
    1. Apply SVD adjustment to entire vocabulary logits for each position
    2. adjusted_logits = ft_logits + λ * delta_z  (entire vocabulary)
    3. Compute log_softmax (entire vocabulary) - compute log probability directly, more stable
    4. Extract log probability of target token
    
    Args:
        delta_z_list: List of delta_z for each position (each shape: [vocab_size])
        ft_logits_list: List of ft_logits for each position (each shape: [vocab_size])
        token_ids_list: List of token IDs for each position
        lambda_scale: Global calibration strength λ
        device: Compute device
    
    Returns:
        logprob: Total log probability for choice (sum of all tokens)
    """
    if len(delta_z_list) == 0:
        return float('-inf')
    
    # Transfer to GPU
    delta_z_gpu = [dz.to(device) for dz in delta_z_list]
    ft_logits_gpu = [fl.to(device) for fl in ft_logits_list]
    
    # ⚠️ Key fix: When λ ≈ 0, use original logits directly, avoid 0 * (-inf) = NaN
    use_lambda = abs(lambda_scale) >= 1e-10
    
    answer_logprob = 0.0
    for delta_z, ft_logits, token_id in zip(delta_z_gpu, ft_logits_gpu, token_ids_list):
        if use_lambda:
            # Apply SVD adjustment to entire vocabulary: adjusted_logits = ft_logits + λ · delta_z
            adjusted_logits = ft_logits + lambda_scale * delta_z
        else:
            # When λ≈0, use original logits directly, avoid 0 * (-inf) = NaN
            adjusted_logits = ft_logits
        
        # Compute log_softmax over entire vocabulary (reference lm-eval, more numerically stable)
        adjusted_log_probs = F.log_softmax(adjusted_logits, dim=-1)
        
        # Extract log probability of target token
        token_logprob = adjusted_log_probs[token_id]
        answer_logprob += token_logprob.item()
    
    # Return total log probability (reference lm-eval, return sum not average)
    return answer_logprob


def compute_choices_logprobs_from_cache(
    delta_z_lists: List[List[torch.Tensor]],
    ft_logits_lists: List[List[torch.Tensor]],
    token_ids_lists: List[List[int]],
    lambda_scale: float,
    device: str = "cuda"
) -> List[float]:
    """
    Batch compute log probabilities for multiple choices (optimized: transfer all data to GPU at once)
    (reference lm-evaluation-harness implementation)
    
    Correct SVD decoding flow (for each position of each choice):
    1. Apply SVD adjustment to entire vocabulary: adjusted_logits = ft_logits + λ * delta_z
    2. Compute log_softmax over entire vocabulary (more numerically stable)
    3. Extract log probability of target token
    
    Args:
        delta_z_lists: List[List[Tensor]] - delta_z list for each choice (each element shape: [vocab_size])
        ft_logits_lists: List[List[Tensor]] - ft_logits list for each choice (each element shape: [vocab_size])
        token_ids_lists: List[List[int]] - token_ids list for each choice
        lambda_scale: Global calibration strength λ
        device: Compute device
    
    Returns:
        List[float] - Total log probability for each choice
    """
    # ⚡ Key optimization: Transfer all tensors for all choices to GPU at once
    # This maximizes PCIe bandwidth utilization and reduces transfer overhead
    
    # Pre-transfer all tensors to GPU (using list comprehension, non-blocking transfer)
    all_delta_z_gpu = [
        [dz.to(device, non_blocking=True) for dz in delta_z_list] if delta_z_list else []
        for delta_z_list in delta_z_lists
    ]
    all_ft_logits_gpu = [
        [fl.to(device, non_blocking=True) for fl in ft_logits_list] if ft_logits_list else []
        for ft_logits_list in ft_logits_lists
    ]
    
    # Synchronize to wait for all transfers to complete (only need one sync)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Now all data is on GPU, quickly compute log probability for each choice
    # ⚠️ Key fix: When λ ≈ 0, use original logits directly, avoid 0 * (-inf) = NaN
    use_lambda = abs(lambda_scale) >= 1e-10
    
    logprobs = []
    for delta_z_gpu, ft_logits_gpu, token_ids_list in zip(all_delta_z_gpu, all_ft_logits_gpu, token_ids_lists):
        if len(delta_z_gpu) == 0:
            logprobs.append(float('-inf'))
            continue
        
        answer_logprob = 0.0
        
        # Fast computation on GPU (no data transfer wait)
        for delta_z, ft_logits, token_id in zip(delta_z_gpu, ft_logits_gpu, token_ids_list):
            if use_lambda:
                # Apply SVD adjustment to entire vocabulary: adjusted_logits = ft_logits + λ · delta_z
                adjusted_logits = ft_logits + lambda_scale * delta_z  # Entire vocabulary [vocab_size]
            else:
                # When λ≈0, use original logits directly, avoid 0 * (-inf) = NaN
                adjusted_logits = ft_logits
            
            # Compute log_softmax over entire vocabulary (reference lm-eval, more numerically stable)
            adjusted_log_probs = F.log_softmax(adjusted_logits, dim=-1)  # [vocab_size]
            
            # Extract log probability of target token
            token_logprob = adjusted_log_probs[token_id]
            answer_logprob += token_logprob.item()
        
        # Return total log probability (reference lm-eval, return sum not average)
        logprobs.append(answer_logprob)
    
    return logprobs


def build_choices_logits_cache(
    choices: List[str],
    prompt: str,
    base_model,
    finetuned_model,
    steering_computer,
    tokenizer,
    device: str,
    method: str
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[List[int]]]:
    """
    Build logits cache for multiple choices
    
    Args:
        choices: List of choices
        prompt: Prompt
        base_model: Original model
        finetuned_model: Fine-tuned model
        steering_computer: SteeringVectorComputer instance
        tokenizer: Tokenizer
        device: Device
        method: SVD method
    
    Returns:
        delta_z_lists: delta_z lists for all choices
        ft_logits_lists: ft_logits lists for all choices
        token_ids_lists: token_ids lists for all choices
    """
    delta_z_lists = []
    ft_logits_lists = []
    token_ids_lists = []
    
    for choice in choices:
        delta_z, ft_logits, token_ids = compute_choice_delta_and_logits(
            prompt, choice, base_model, finetuned_model, steering_computer,
            tokenizer, device, method
        )
        delta_z_lists.append(delta_z)
        ft_logits_lists.append(ft_logits)
        token_ids_lists.append(token_ids)
    
    return delta_z_lists, ft_logits_lists, token_ids_lists


def predict_from_logprobs(logprobs: List[float], choices: List[str]) -> str:
    """
    Predict answer based on log probabilities
    
    Args:
        logprobs: List of log probabilities for each choice
        choices: List of choices
    
    Returns:
        predicted_choice: Choice with highest probability
    """
    if len(logprobs) == 0 or len(choices) == 0:
        return ""
    
    best_idx = np.argmax(logprobs)
    return choices[best_idx]


class QuestionStoppingCriteria(StoppingCriteria):
    """
    Truncate generation at specified stop string to prevent model from repeating question stem.
    """
    
    def __init__(
        self,
        tokenizer,
        stop_string: str = "Question:",
        initial_input_length: int = 0
    ):
        """
        Args:
            tokenizer: Tokenizer for decoding tokens
            stop_string: Stop string to detect (default "Question:")
            initial_input_length: Initial input length (number of prompt tokens)
        """
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.initial_input_length = initial_input_length
        # Cache decoded text to avoid repeated decoding
        # Initialize to initial_input_length, only check generated part, not prompt
        self.cached_text = ""
        self.cached_length = initial_input_length
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if generation should stop
        
        Args:
            input_ids: Current generated token IDs [batch_size, seq_len]
            scores: Current logits [batch_size, vocab_size]
        
        Returns:
            should_stop: True if stop string detected
        """
        # Only check first batch (usually batch_size=1)
        current_ids = input_ids[0]
        current_length = current_ids.shape[0]
        
        # Only decode newly generated part (from last cached position)
        if current_length > self.cached_length:
            # Extract new tokens (from last cached position to current position)
            new_tokens = current_ids[self.cached_length:current_length]
            if new_tokens.shape[0] > 0:
                # Decode new tokens and append to cache
                new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                self.cached_text += new_text
            self.cached_length = current_length
        
        # Check if contains stop string
        if self.stop_string in self.cached_text:
            return True
        
        return False


def create_question_stopping_criteria(
    tokenizer,
    stop_string: Optional[str] = None,
    dataset_name: Optional[str] = None,
    initial_input_length: int = 0
) -> Optional[StoppingCriteriaList]:
    """
    Select appropriate stop string based on dataset and construct StoppingCriteriaList.
    """
    resolved = stop_string if stop_string is not None else get_dataset_stop_string(dataset_name)
    
    if resolved:
        criteria = QuestionStoppingCriteria(
            tokenizer=tokenizer,
            stop_string=resolved,
            initial_input_length=initial_input_length
        )
        return StoppingCriteriaList([criteria])
    return None


if __name__ == "__main__":
    # Test code
    print("Evaluation utility functions library loaded successfully")
    
    # Test answer extraction
    print("\nTest answer extraction:")
    test_cases = [
        "The answer is #### $16",
        "Step by step... #### 1,000",
        "Result: #### 25%",
        "No separator here, just 42"
    ]
    for test in test_cases:
        extracted = extract_answer_from_generation(test)
        print(f"  '{test}' -> '{extracted}'")
    
    # Test answer normalization
    print("\nTest answer normalization:")
    test_answers = ["$1,000", "16.0", "25%", "42"]
    for ans in test_answers:
        normalized = normalize_numerical_answer(ans)
        print(f"  '{ans}' -> '{normalized}'")
    
    # Test answer comparison
    print("\nTest answer comparison:")
    test_comparisons = [
        ("16", "16", True),
        ("1000", "1,000", True),
        ("16.0", "16", True),
        ("$25", "25", True),
        ("42%", "42", True),
        ("The answer is 16", "16", True),
        ("16", "17", False),
    ]
    for pred, truth, expected in test_comparisons:
        result = compare_numerical_answers(pred, truth)
        status = "✓" if result == expected else "✗"
        print(f"  {status} compare('{pred}', '{truth}') = {result} (expected: {expected})")

