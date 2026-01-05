"""AQUA-RAT Generative Evaluator"""
import re

from .generative_reasoning_base import GenerativeReasoningEvaluator


LETTER_PATTERN = re.compile(r"\b([A-E])\b", re.IGNORECASE)


class AquaRATEvaluator(GenerativeReasoningEvaluator):
    def __init__(self, tokenizer, device: str = "cuda", model_name: str = None):
        super().__init__(
            tokenizer,
            device,
            dataset_name="aqua_rat",
            final_answer_key="final_answer",
            model_name=model_name
        )

    def extract_final_answer(self, generated_text: str) -> str:
        segment = generated_text
        if "####" in generated_text:
            segment = generated_text.split("####")[-1]
        match = LETTER_PATTERN.search(segment)
        if match:
            return match.group(1).upper()
        # Fall back to last non-empty character
        segment = segment.strip()
        return segment[-1].upper() if segment else ""

    def normalize_answer(self, answer: str) -> str:
        if not answer:
            return ""
        answer = answer.strip().upper()
        if answer and answer[0] in "ABCDE":
            return answer[0]
        match = LETTER_PATTERN.search(answer)
        if match:
            return match.group(1).upper()
        return answer[:1]

    def check_correctness(self, prediction: str, ground_truth: str) -> bool:
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)


def evaluate_with_generation(
    model,
    tokenizer,
    data,
    device="cuda",
    max_samples=None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    do_sample: bool = False
):
    evaluator = AquaRATEvaluator(tokenizer, device)
    return evaluator.evaluate_dataset(
        model,
        data,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample
    )

