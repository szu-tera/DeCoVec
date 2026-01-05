# Experiment Reproduction Guide

This guide provides step-by-step instructions to reproduce all results from the ACL 2026 paper.

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Main Experiments](#main-experiments)
4. [Ablation Studies](#ablation-studies)
5. [Analysis Experiments](#analysis-experiments)

## Environment Setup

### Hardware Requirements

- GPU: NVIDIA GPU with ≥24GB VRAM (for 7B-9B models)
- RAM: ≥32GB
- Storage: ≥100GB free space

### Software Requirements

```bash
# Python environment
python >= 3.8
pytorch >= 2.0
transformers >= 4.30
sentence-transformers >= 2.2

# Install all dependencies
pip install -r requirements.txt
```

### Model Download

Models will be automatically downloaded on first run, or download manually:

```bash
bash download_models.sh
```

This downloads:
- Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B
- Llama-2-7B, Llama-3-8B
- Yi-6B
- Gemma-2-9B

## Data Preparation

All datasets are preprocessed and ready to use in `data/` directory:

```bash
data/
├── truthfulqa/              # TruthfulQA dataset
│   ├── truthfulqa_qa.json
│   └── truthfulqa_qa_calibration.json
├── math500/                 # Math-500 dataset
│   ├── math500.json
│   └── math500_calibration.json
└── aqua_rat/                # AQUA-RAT dataset
    ├── aqua_rat.json
    └── aqua_rat_calibration.json
```

To regenerate datasets (optional):

```bash
# TruthfulQA
python data/prepare_truthfulqa_qa.py

# Math-500
python data/prepare_math500.py

# AQUA-RAT
python data/prepare_aqua_rat.py
```

## Main Experiments

### Table 2: Main Results

Reproduce complete results for all models and datasets.

#### TruthfulQA

```bash
# Qwen2.5-7B
python run_decovec.py --dataset truthfulqa --model qwen2.5-7b --mode test_scale --lambda_values 1.0 --n_shot 20 --icl_methods kate

# Llama-2-7B
python run_decovec.py --dataset truthfulqa --model llama2-7b --mode test_scale --lambda_values 1.0 --n_shot 20 --icl_methods kate

# Llama-3-8B
python run_decovec.py --dataset truthfulqa --model llama-3-8b --mode test_scale --lambda_values 1.0 --n_shot 20 --icl_methods kate
```

#### Math-500

```bash
python run_decovec.py --dataset math500 --model qwen2.5-7b --mode test_scale --lambda_values 0.2,0.4,0.6,0.8,1.0 --n_shot 2 --icl_methods kate
# ... (repeat for other models)
```

#### AQUA-RAT

```bash
python run_decovec.py --dataset aqua_rat --model qwen2.5-7b --mode test_scale --lambda_values 0.2,0.4,0.6,0.8,1.0 --n_shot 2 --icl_methods kate
# ... (repeat for other models)
```

**Expected Runtime**: ~2-4 hours per model-dataset combination on A100 GPU

### Different Demonstration Selection Strategies

Test all selection strategies (Random, KATE, BM25):

```bash
# TruthfulQA with all strategies
for strategy in random_icl kate bm25; do
    python run_decovec.py \
        --dataset truthfulqa \
        --model qwen2.5-7b \
        --mode test_scale \
        --lambda_values 1.0 \
        --n_shot 15 \
        --icl_methods $strategy
done
```

## Ablation Studies

### Table 3: Scaling Factor (λ) Ablation

Test different λ values to analyze the impact of task vector strength.

```bash
# TruthfulQA
python run_decovec.py \
  --mode test_scale \
  --dataset truthfulqa \
  --lambda_values 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0 \
  --n_shot 20 \
  --icl_methods kate

# Math-500
python run_decovec.py \
  --mode test_scale \
  --dataset math500 \
  --lambda_values 0.2,0.4,0.6,0.8,1.0 \
  --n_shot 2 \
  --icl_methods kate

# AQUA-RAT
python run_decovec.py \
  --mode test_scale \
  --dataset aqua_rat \
  --lambda_values 0.2,0.4,0.6,0.8,1.0 \
  --n_shot 2 \
  --icl_methods kate
```

Manual testing:
```bash
for lambda in 0.5 1.0 1.5 2.0; do
    python run_decovec.py \
        --dataset truthfulqa \
        --model qwen2.5-7b \
        --mode test_scale \
        --lambda_values $lambda \
        --n_shot 20 \
        --icl_methods kate
done
```

**Expected Results**: Performance peaks around λ=1.0-1.5

### Table 4: Demonstration Pool Size Ablation

Test impact of calibration pool size on performance.

```bash
# TruthfulQA pool ablation (using pool_ablation datasets)
python run_decovec.py \
  --mode test_scale \
  --dataset truthfulqa_pool_ablation \
  --lambda_values 1.0 \
  --n_shot 20 \
  --icl_methods kate
```

Manual testing:
```bash
for pool_size in 50 100 200 500; do
    python run_decovec.py \
        --dataset truthfulqa_pool_ablation \
        --model qwen2.5-7b \
        --mode test_scale \
        --lambda_values 1.0 \
        --n_shot 20 \
        --calibration_pool_size $pool_size \
        --icl_methods kate
done
```

### Table 5: Number of Shots Ablation

Test different numbers of ICL demonstrations.

```bash
# Run shot ablation
for n_shot in 10 20 30 40 50 60 70 80 90 100; do
    python run_decovec.py \
        --dataset truthfulqa \
        --model llama2-7b \
        --mode test_scale \
        --lambda_values 1.0 \
        --n_shot $n_shot \
        --icl_methods kate
done
```

## Analysis Experiments

### Figure 3: Example Order Sensitivity

Analyze the impact of demonstration ordering.
```bash
# Ordered (similarity-based)
python run_decovec.py --dataset truthfulqa --model qwen2.5-7b --mode test_scale --lambda_values 1.0 --n_shot 20 --icl_methods kate --example_order ordered

# Reversed
python run_decovec.py --dataset truthfulqa --model qwen2.5-7b --mode test_scale --lambda_values 1.0 --n_shot 20 --icl_methods kate --example_order reverse

# Random (with seed)
python run_decovec.py --dataset truthfulqa --model qwen2.5-7b --mode test_scale --lambda_values 1.0 --n_shot 20 --icl_methods kate --example_order random --example_order_seed 42
```

### Error Analysis

Analyze error types on mathematical reasoning tasks:

```bash
# Generate detailed outputs for analysis
python run_decovec.py \
    --dataset math500 \
    --model qwen2.5-7b \
    --mode test_scale \
    --lambda_values 1.0 \
    --n_shot 10 \
    --icl_methods kate \
    --save_outputs
```

### Task Vector Visualization

Visualize the learned task vectors:

```bash
python analysis/visualize_task_vector.py \
    --model qwen2-7b \
    --dataset math500
```

## Variance Analysis

Run experiments with multiple seeds:
```bash
for seed in 42 123 456 789 999; do
    python run_decovec.py \
        --dataset truthfulqa \
        --model qwen2.5-7b \
        --mode test_scale \
        --lambda_values 1.0 \
        --n_shot 15 \
        --icl_methods kate \
        --example_order random \
        --example_order_seed $seed
done
```

## Troubleshooting

### Issue 1: Model Download Fails

```bash
# Set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### Issue 2: Import Errors

```bash
# Verify directory structure
ls decovec/  # Should show decovec_core.py, etc.

# Test imports
python -c "from decovec.decovec_core import TaskVectorBuilder"
python -c "from decovec.scale_tester import ScaleTester"
python -c "from decovec.experiment_manager import ExperimentManager"
```

### Issue 3: Slow Generation

Enable Flash Attention (if available):
```bash
pip install flash-attn
```

## Questions?

- Check [API.md](API.md) for detailed API documentation
- Open an issue on GitHub
- Contact the authors

---

**Estimated Total Time**: 2-3 days for complete reproduction on single A100 GPU
