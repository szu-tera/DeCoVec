"""
Prompt loading and formatting utilities
Unified management of dataset prompt templates
"""
import json
import os
from typing import Dict, List, Any


def load_prompt_config(dataset_name: str) -> Dict:
    """
    Load prompt configuration for a dataset
    
    Args:
        dataset_name: Dataset name (supports: "truthfulqa", "math500", "aqua_rat", and other custom datasets)
    
    Returns:
        prompt_config: Prompt configuration dictionary
    """
    # Get data directory path
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build prompt.json file path
    prompt_file = os.path.join(data_dir, dataset_name, "prompt.json")
    
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt configuration file not found: {prompt_file}")
    
    # Load configuration
    with open(prompt_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def format_choices(choices: List[str], labels: List[str], choice_template: str = "{label}. {text}") -> str:
    """
    Format choice list (for multiple choice questions)
    
    Args:
        choices: Choice text list
        labels: Choice label list
        choice_template: Choice format template
    
    Returns:
        formatted_choices: Formatted choice string
    """
    formatted = []
    for label, text in zip(labels, choices):
        formatted.append(choice_template.format(label=label, text=text))
    return "\n".join(formatted)


def format_zero_shot_prompt(data_item: Dict, prompt_config: Dict, dataset_name: str = None) -> str:
    """
    Generate Zero-shot prompt
    
    Args:
        data_item: Data sample
        prompt_config: Prompt configuration
        dataset_name: Dataset name (for special handling)
    
    Returns:
        prompt: Complete zero-shot prompt
    """
    # Get template
    template = prompt_config["templates"]["zero_shot"]
    
    # Prepare placeholder replacement dictionary
    replacements = dict(data_item)
    replacements.setdefault("question", data_item.get("question", ""))
    
    # Special handling: CommonsenseQA needs formatted choices
    if "choices" in data_item and "labels" in data_item:
        choice_template = prompt_config.get("choice_template", "{label}. {text}")
        replacements["choices"] = format_choices(
            data_item["choices"],
            data_item["labels"],
            choice_template
        )
    
    # Special handling: BoolQ needs to add instruction
    if "instruction" in prompt_config:
        instruction = prompt_config["instruction"]
        prompt = instruction + " " + template.format(**replacements)
    else:
        prompt = template.format(**replacements)
    
    # Special handling: TruthfulQA/ASDiv etc. need instruction and fixed examples
    if "few_shot_examples" in prompt_config:
        instruction = prompt_config.get("instruction", "")
        few_shot_examples = prompt_config["few_shot_examples"]
        
        # Build fixed examples part (use format_icl_demo to support different dataset formats)
        examples_text = ""
        for example in few_shot_examples:
            examples_text += format_icl_demo(example, prompt_config, dataset_name)
        
        # Complete prompt: instruction + fixed examples + query
        prompt = instruction + examples_text + template.format(**replacements)
    
    return prompt


def format_icl_demo(data_item: Dict, prompt_config: Dict, dataset_name: str = None) -> str:
    """
    Format ICL demonstration example
    
    Args:
        data_item: Data sample
        prompt_config: Prompt configuration
        dataset_name: Dataset name
    
    Returns:
        demo: Formatted demonstration text
    """
    # Get template
    template = prompt_config["templates"]["icl_demo"]
    
    # Prepare placeholder replacement dictionary
    replacements = dict(data_item)
    replacements.setdefault("question", data_item.get("question", ""))
    
    # Special handling: news_factor dataset uses full answer text in ICL Demo (for MC1 loglikelihood computation)
    if dataset_name == "news_factor":
        # Use answer_text or best_answer (full text), not answer_label (letter)
        replacements["answer"] = data_item.get("answer_text", data_item.get("best_answer", data_item.get("answer", "")))
    else:
        # Other datasets use answer or best_answer
        replacements.setdefault("answer", data_item.get("answer", data_item.get("best_answer", "")))
    
    # Special handling: CommonsenseQA needs formatted choices
    if "choices" in data_item and "labels" in data_item:
        choice_template = prompt_config.get("choice_template", "{label}. {text}")
        replacements["choices"] = format_choices(
            data_item["choices"],
            data_item["labels"],
            choice_template
        )
    
    return template.format(**replacements)


def format_icl_query(data_item: Dict, prompt_config: Dict, dataset_name: str = None) -> str:
    """
    Format ICL query (similar to zero_shot, but used in ICL context)
    
    Args:
        data_item: Data sample
        prompt_config: Prompt configuration
        dataset_name: Dataset name
    
    Returns:
        query: Formatted query text
    """
    # Get template
    template = prompt_config["templates"]["icl_query"]
    
    # Prepare placeholder replacement dictionary
    replacements = dict(data_item)
    replacements.setdefault("question", data_item.get("question", ""))
    
    # Special handling: CommonsenseQA needs formatted choices
    if "choices" in data_item and "labels" in data_item:
        choice_template = prompt_config.get("choice_template", "{label}. {text}")
        replacements["choices"] = format_choices(
            data_item["choices"],
            data_item["labels"],
            choice_template
        )
    
    return template.format(**replacements)


def construct_icl_prompt(
    examples: List[Dict],
    query_item: Dict,
    prompt_config: Dict,
    dataset_name: str,
    include_query_answer: bool = False,
    query_answer: str = None
) -> str:
    """
    Construct complete ICL prompt
    
    Args:
        examples: ICL example list
        query_item: Query sample
        prompt_config: Prompt configuration
        dataset_name: Dataset name
        include_query_answer: Whether to include query answer (for calibration)
        query_answer: Query answer
    
    Returns:
        prompt: Complete ICL prompt
    """
    prompt_parts = []
    
    # 1. Add instruction (if available)
    if "instruction" in prompt_config:
        prompt_parts.append(prompt_config["instruction"])
    
    # 2. Add fixed examples (TruthfulQA)
    if "few_shot_examples" in prompt_config:
        for example in prompt_config["few_shot_examples"]:
            demo_text = format_icl_demo(example, prompt_config, dataset_name)
            prompt_parts.append(demo_text)
    
    # 3. Add dynamically selected examples
    for example in examples:
        demo_text = format_icl_demo(example, prompt_config, dataset_name)
        prompt_parts.append(demo_text)
    
    # 4. Add query
    if include_query_answer and query_answer:
        # Include answer (for calibration)
        query_with_answer = query_item.copy()
        query_with_answer["answer"] = query_answer
        query_text = format_icl_demo(query_with_answer, prompt_config, dataset_name)
    else:
        # Exclude answer (for inference)
        query_text = format_icl_query(query_item, prompt_config, dataset_name)
    
    prompt_parts.append(query_text)
    
    # 5. Concatenate (BoolQ uses spaces, others use newlines)
    if dataset_name == "boolq":
        # BoolQ's instruction needs a space after it
        if "instruction" in prompt_config:
            instruction = prompt_parts[0]
            rest = prompt_parts[1:]
            prompt = instruction + " " + " ".join(rest)
        else:
            prompt = " ".join(prompt_parts)
    else:
        # Use empty string to concatenate (each part already has newline at end)
        prompt = "".join(prompt_parts)
    
    return prompt


if __name__ == "__main__":
    # Test code for TruthfulQA (paper dataset)
    print("Testing Prompt Loader")
    
    # Test TruthfulQA
    print("\n=== TruthfulQA ===")
    truthfulqa_config = load_prompt_config("truthfulqa")
    truthfulqa_item = {
        "question": "What happens if you eat watermelon seeds?",
        "best_answer": "The watermelon seeds pass through your digestive system."
    }
    print("Zero-shot:", format_zero_shot_prompt(truthfulqa_item, truthfulqa_config)[:200] + "...")
    print("ICL Demo:", format_icl_demo(truthfulqa_item, truthfulqa_config))
    
    print("\n✓ Prompt Loader test complete")

