"""
Simplified configuration module with paper-relevant parameters only.
Used by ExperimentManager and internal modules.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimpleConfig:
    """DeCoVec experiment configuration (paper parameters only)."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2-7B"
    model_path: str = "checkpoints/qwen/Qwen2-7B"
    device: str = "cuda:0"
    torch_dtype: str = "float16"  # "float16" or "bfloat16"
    
    # Embedding model (used for KATE example selection)
    emb_model_path: str = "checkpoints/emb_models/all-MiniLM-L6-v2"
    emb_model_name: str = "all-MiniLM-L6-v2"  # fallback
    
    # Evaluation configuration
    batch_size: int = 8
    temperature: float = 0.0  # Temperature for generative tasks (0=greedy)
    max_samples: Optional[int] = None  # None = evaluate all samples
    results_dir: str = "results"
    
    # Delta_z mask: mask tokens with prob < threshold * max_prob (softmax on ICL logits)
    delta_z_use_mask: bool = True
    delta_z_mask_threshold: float = 0.1  # mask if prob < 0.1 * max_prob
    
    # Delta_z mask configuration (mask low-prob tokens in steering vector)
    delta_z_use_mask: bool = True  # Default: mask tokens with prob < threshold * max_prob
    delta_z_mask_threshold: float = 0.1  # Mask if prob < 0.1 * max_prob
    
    # Dataset configuration
    data_dir: str = "data"
    seed: int = 42  # Default random seed
    use_full_test_set: bool = True  # Use full test set
    
    # Dataset-specific seeds (paper settings)
    dataset_seeds: dict = None
    
    # Experiment name
    experiment_name: str = "decovec_experiment"
    
    def __post_init__(self):
        """Initialize dataset seed mapping."""
        if self.dataset_seeds is None:
            self.dataset_seeds = {
                "truthful_qa": 44,
                "truthfulqa": 44,
                "boolq": 42,
                "commonsense_qa": 42,
                "gsm8k": 42,
                "math500": 42,
                "aqua_rat": 42,
            }
        
        # Ensure required directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_seed(self, dataset_name: str) -> int:
        """Get the seed for a dataset."""
        return self.dataset_seeds.get(dataset_name, self.seed)


# Global configuration instance
_config = SimpleConfig()


def get_config():
    """Return the global configuration."""
    return _config


def set_config(new_config: SimpleConfig):
    """Set the global configuration."""
    global _config
    _config = new_config
