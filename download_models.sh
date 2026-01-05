#!/bin/bash
# DeCoVec Model Download Script
# Download models to local using ModelScope

# ============================================
# Qwen2 Series (0.5B - 7B)
# ============================================

# Qwen2-0.5B
# modelscope download --model Qwen/Qwen2-0.5B --local_dir checkpoints/qwen/Qwen2-0.5B

# Qwen2-1.5B
# modelscope download --model Qwen/Qwen2-1.5B --local_dir checkpoints/qwen/Qwen2-1.5B

# Qwen2-7B (main model used in paper, recommended to download first)
modelscope download --model Qwen/Qwen2-7B --local_dir checkpoints/qwen/Qwen2-7B

# ============================================
# Other Models
# ============================================

# Yi-6B
# modelscope download --model 01ai/Yi-6B --local_dir checkpoints/yi/Yi-6B

# Llama-2-7B
# modelscope download --model shakechen/Llama-2-7b-hf --local_dir checkpoints/llama/Llama-2-7b-hf

# Llama-3-8B
# modelscope download --model LLM-Research/Meta-Llama-3-8B --local_dir checkpoints/llama/Meta-Llama-3-8B

# Gemma-2-9B
# modelscope download --model google/gemma-2-9b --local_dir checkpoints/gemma/gemma-2-9b

# ============================================
# Embedding Models (for KATE example selection)
# ============================================

# all-MiniLM-L6-v2 (recommended)
modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --local_dir ./checkpoints/emb_models/all-MiniLM-L6-v2

echo "✓ Embedding model download complete"