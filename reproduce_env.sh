#!/bin/bash
# Sentinel H100 Rapid Deployment Script
# Purpose: Rebuild the H100 environment in <10 mins

echo "🚀 Starting Sentinel Environment Recovery..."

sudo apt update && sudo apt install -y python3-pip python3-venv screen

# Create and Activate Venv
python3 -m venv ~/venv_sentinel
source ~/venv_sentinel/bin/activate

# Install AI Stack
pip install --upgrade pip
pip install torch transformers accelerate bitsandbytes sentencepiece huggingface_hub

# Compile H100 Optimized Kernels (This takes the longest)
echo "⚡ Compiling Flash-Attention 2 for sm_90..."
pip install flash-attn --no-build-isolation

echo "✅ Environment Ready. Please run 'huggingface-cli login' next."
