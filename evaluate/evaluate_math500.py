"""Math-500 Generative Evaluator"""
from .generative_reasoning_base import GenerativeReasoningEvaluator, evaluate_with_generation as base_eval_with_generation


class Math500Evaluator(GenerativeReasoningEvaluator):
    def __init__(self, tokenizer, device: str = "cuda", model_name: str = None):
        super().__init__(tokenizer, device, dataset_name="math500", final_answer_key="final_answer", model_name=model_name)


def evaluate_with_generation(
    model,
    tokenizer,
    data,
    device="cuda",
    max_samples=None,
    max_new_tokens: int = 1024,  # Output at most 1024 tokens
    temperature: float = 0.7,
    do_sample: bool = False
):
    return base_eval_with_generation(
        model,
        tokenizer,
        data,
        dataset_name="math500",
        final_answer_key="final_answer",
        device=device,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample
    )

