# DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning

[![Anonymous Submission](https://img.shields.io/badge/Submission-Anonymous-red)](https://openreview.net/)
[![Python 3.10+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Anonymous submission for peer review. Author information withheld.**

## 🌟 Overview

**DeCoVec** is a training-free and non-invasive framework for constructing task vectors directly in the **decoding space**. Unlike traditional methods that operate in model weights or internal activations, DeCoVec:

- ✨ **Training-free**: No gradient updates or fine-tuning required
- 🔓 **Non-invasive**: No modification to model parameters or internal states
- 🎯 **Effective**: Consistent performance gains across knowledge-intensive and reasoning tasks
- 🚀 **Plug-and-play**: Works seamlessly with any pre-trained LLM

### Key Idea

DeCoVec constructs task vectors by contrasting the output logit distributions between few-shot ICL context and zero-shot context:

```
v_T^t = z_icl^t - z_zs^t
```

Then steers generation by injecting the task vector into the decoding process:

```
z_tilde^t = z_de^t + λ · v_T^t
```

## 📊 Main Results

Performance improvements over few-shot baselines across different models and tasks:

| Model | TruthfulQA (Avg Δ) | Math-500 (Avg Δ) | AQUA-RAT (Avg Δ) |
| :--- | :--- | :--- | :--- |
| **Qwen2-0.5B** | +1.58% | +0.88% | +4.14% |
| **Qwen2-1.5B** | +1.20% | +2.44% | +4.53% |
| **Qwen2-7B** | +1.47% | +2.73% | +2.07% |
| **Yi-6B** | +2.41% | +1.21% | +2.56% |
| **Llama-2-7B** | +2.31% | +3.62% | +1.28% |
| **Llama-3-8B** | +1.41% | +0.98% | +3.44% |
| **Gemma-2-9B** | +1.22% | +3.22% | +2.56% |

See paper Table 2 for complete results.

## 🔧 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/DeCoVec
cd DeCoVec

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional, will auto-download on first run)
bash download_models.sh
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run DeCoVec on TruthfulQA with Qwen2-7B
python run_decovec.py \
    --dataset truthfulqa \
    --model qwen2-7b \
    --mode test_scale \
    --icl_method kate \
    --n_shot 15

# Run on Math-500
python run_decovec.py \
    --dataset math500 \
    --model qwen2-7b \
    --mode test_scale \
    --icl_method kate \
    --n_shot 2
```

### Configuration

Edit `simple_config.py` to customize:
- Model selection
- Dataset paths
- Hyperparameters (λ, etc.)

Key hyperparameters:
- `lambda_scale`: Scaling factor for task vector
- `n_shot`: Number of ICL demonstrations (15 for TruthfulQA, 10 for reasoning tasks)

## 📁 Project Structure

```
DeCoVec/
├── decovec/                    # Core implementation
│   ├── decovec_core.py        # Main algorithm
│   ├── decovec_processor.py   # LogitsProcessor for HuggingFace
│   ├── task_vector_builder.py # Task vector construction
│   ├── demonstration_sampler.py # ICL example selection
│   └── ...
├── data/                       # Datasets
│   ├── truthfulqa/
│   ├── math500/
│   └── aqua_rat/
├── evaluate/                   # Evaluation metrics
├── docs/                       # Documentation
├── run_decovec.py             # Main entry point
├── simple_config.py           # Configuration
└── README.md
```

## 🔬 Reproduce Paper Results

### Main Experiments (Table 2)

```bash
# Test different λ values on TruthfulQA
python run_decovec.py \
    --mode test_scale \
    --dataset truthfulqa \
    --model qwen2-7b \
    --lambda_values 1.0 \
    --n_shot 20

# Test on Math-500
python run_decovec.py \
    --mode test_scale \
    --dataset math500 \
    --model qwen2-7b \
    --lambda_values 0.2,0.4,0.6,0.8,1.0 \
    --n_shot 2
```

## 📖 Documentation

- [API Documentation](docs/API.md)

## 🎯 Key Components

### Core Algorithm

The main implementation is in `decovec/decovec_core.py`:

```python
from decovec.decovec_core import TaskVectorBuilder

# Initialize task vector builder
builder = TaskVectorBuilder()

# Compute task vector (Eq. 7)
delta_z = builder.compute_delta_z(
    logits_base,      # Zero-shot logits (z_zs)
    logits_finetuned  # ICL logits (z_icl)
)
```

### Correspondence with Paper

| Paper Notation | Code Location | Description |
|---------------|---------------|-------------|
| Eq. 7 | `decovec_core.py::TaskVectorBuilder.compute_delta_z()` | Compute v_T = z_icl - z_zs |
| λ (lambda) | `scale_tester.py::test_lambda_values()` | Scaling factor for task vector |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> **Note**: This is an anonymous submission. Author information and acknowledgments will be added upon acceptance.

