# API Documentation

## Core Classes

### TaskVectorBuilder

Builds task vectors in the decoding space.

**Location**: `decovec/decovec_core.py`

**Paper Reference**: Implements Equation 7: `v_T^t = z_icl^t - z_zs^t`

```python
from decovec.decovec_core import TaskVectorBuilder

builder = TaskVectorBuilder()

# Compute task vector (delta_z)
delta_z = builder.compute_delta_z(
    logits_base,        # Zero-shot logits (z_zs)
    logits_finetuned,   # ICL logits (z_icl)
)
```

**Methods**:
- `compute_delta_z()`: Computes task vector with automatic centering (Equation 7)

### ScaleTester

Tests different lambda scaling values for task vectors.

**Location**: `decovec/scale_tester.py`

**Paper Reference**: Manual lambda hyperparameter search (Section 4.2)

```python
from decovec.scale_tester import ScaleTester

# ScaleTester is created through ExperimentManager
# Use run_decovec.py directly for testing
# Example:
# python run_decovec.py \
#   --mode test_scale \
#   --dataset aqua_rat \
#   --lambda_values 0.5,1.0,1.5,2.0 \
#   --n_shot 10 \
#   --icl_methods kate
```

### DemonstrationSampler

Selects ICL demonstrations using various strategies.

**Location**: `decovec/demonstration_sampler.py`

**Paper Reference**: Implements sampler S in Section 3.1

```python
from decovec.demonstration_sampler import DemonstrationSampler
from sentence_transformers import SentenceTransformer

# Load embedding model
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

sampler = DemonstrationSampler(
    emb_model=emb_model,
    n_shot=15,
    dataset_type="truthfulqa",
    selection_mode="kate",  # Options: "kate", "random_icl", "bm25"
    example_order="ordered"  # Options: "ordered", "reverse", "random"
)

# Build kNN index for calibration data
embeddings, knn_index = sampler.build_knn_index(calibration_data)

# Sample demonstrations for a query
examples = sampler.select_demonstrations(
    query_item=query_item,
    calibration_data=calibration_data
)
```

### DeCoVecLogitsProcessor

HuggingFace LogitsProcessor for integration with `model.generate()`.

**Location**: `decovec/decovec_processor.py`

**Paper Reference**: Implements Equations 9-10

```python
from decovec.decovec_processor import DeCoVecLogitsProcessor

processor = DeCoVecLogitsProcessor(
    model=model,
    tokenizer=tokenizer,
    zero_shot_prompt=zs_prompt,
    icl_prompt=icl_prompt,
    lambda_scale=1.0,
    steering_computer=steering_computer,
    device="cuda"
)

# Use with HuggingFace generate
outputs = model.generate(
    input_ids,
    logits_processor=[processor],
    max_new_tokens=512
)
```

## Configuration

### SimpleConfig

Main configuration class for DeCoVec experiments.

**Location**: `simple_config.py`

```python
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
    
    # Dataset configuration
    data_dir: str = "data"
    seed: int = 42  # Default random seed
    use_full_test_set: bool = True  # Use full test set
    dataset_seeds: dict = None  # Dataset-specific seeds
    experiment_name: str = "decovec_experiment"
```

## Key Functions

### Task Vector Construction

```python
def build_task_vector(z_icl, z_zs):
    """
    Build task vector in decoding space (Eq. 8)
    
    Args:
        z_icl: Few-shot ICL logits (Eq. 7), shape [vocab_size]
        z_zs: Zero-shot logits (Eq. 6), shape [vocab_size]
    
    Returns:
        task_vector: v_T in R^{|V|}, shape [vocab_size]
    """
    return z_icl - z_zs
```

### Steering

```python
def apply_steering(z_de, task_vector, lambda_scale):
    """
    Apply task vector steering (Eq. 10)
    
    Args:
        z_de: Base decoding logits (Eq. 9), shape [vocab_size]
        task_vector: Task vector v_T, shape [vocab_size]
        lambda_scale: Scaling factor (lambda)
    
    Returns:
        z_tilde: Task-oriented logits, shape [vocab_size]
    """
    return z_de + lambda_scale * task_vector
```

## Usage Examples

### Example 1: Testing Lambda Values

```bash
# Test multiple λ values using command line
python run_decovec.py \
  --mode test_scale \
  --dataset aqua_rat \
  --lambda_values 0.2,0.4,0.6,0.8,1.0 \
  --n_shot 2 \
  --icl_methods kate

```

### Example 2: Computing Task Vectors

```python
from decovec.decovec_core import TaskVectorBuilder
import torch

# Initialize task vector builder
builder = TaskVectorBuilder()

# Prepare logits (from model forward pass)
logits_zs = model(zero_shot_input).logits[0, -1, :]   # [vocab_size]
logits_icl = model(icl_input).logits[0, -1, :]        # [vocab_size]

# Compute task vector (paper equation 7)
delta_z = builder.compute_delta_z(
    logits_zs,
    logits_icl
)

# Apply with manual lambda (paper equations 9-10)
lambda_scale = 1.0
adjusted_logits = logits_icl + lambda_scale * delta_z
```

## Data Format

### Input Format

```python
{
    "question": "What is the capital of France?",
    "choices": ["Paris", "London", "Berlin", "Madrid"],
    "answer": "Paris"
}
```

### Demonstration Format

```python
demonstrations = [
    {
        "question": "What is 2+2?",
        "answer": "4"
    },
    {
        "question": "What is the capital of UK?",
        "answer": "London"
    }
]
```

## Evaluation Metrics

### TruthfulQA

- MC1: Single-choice accuracy
- MC2: Multi-choice normalized probability
- MC3: Multi-choice accuracy

### Math-500 & AQUA-RAT

- Accuracy: Exact match accuracy

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `decovec/` directory exists
   ```bash
   ls decovec/  # Should contain decovec_core.py, scale_tester.py, etc.
   ```

2. **CUDA Out of Memory**: Reduce batch size or use smaller models

3. **Slow Generation**: Use GPU acceleration and optimize batch size

## Advanced Configuration

See [simple_config.py](../simple_config.py) for all available options including:
- Model selection
- Dataset paths
- Evaluation settings
- Device configuration

## Contributing

When adding new features:
1. Follow the existing naming conventions
2. Add docstrings with paper equation references
3. Update this API documentation
4. Add tests if applicable
