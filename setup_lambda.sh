#!/bin/bash

# --- Sentinel Lambda A100 Setup Script ---
# This script prepares a fresh Ubuntu instance for High-Fidelity Latent Monitoring.

echo "🚀 Starting Sentinel Infrastructure Setup..."

# 1. Update System and Install Base Dependencies
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential python3-pip git-lfs htop nvtop

# 2. Install High-Performance Python stack
# We use specific versions to ensure Pareto math (KDE) remains stable
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install transformers accelerate bitsandbytes scipy numpy pandas matplotlib

# 3. Install the AI Safety Stack
# UK AI Safety Institute's Framework
pip3 install inspect-ai 

# 4. A100 SPECIAL: Flash Attention (Critical for Llama-3-8B Latent Extraction)
# We install 'ninja' first to prevent the build from hanging for 20 minutes
pip3 install packaging ninja
pip3 install flash-attn --no-build-isolation

# 5. Initialize the Repository Structure (Just in case)
mkdir -p data/raw data/processed results logs experiments/tasks experiments/solvers

# 6. Verify GPU Accessibility
echo "🔍 Verifying GPU Status..."
python3 -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}'); print(f'✅ Device: {torch.cuda.get_device_name(0)}')"

echo "✨ Setup Complete. Team Sentinel is ready for deployment."