# Day 1 Setup Script for Sentinel
# Run this the moment you log into Lambda!

echo "--- Initializing Sentinel Research Environment ---"

# 1. Install necessary AI libraries
# Added scikit-learn for your DeceptionProbe and inspect-ai for the OpenAI tasks
pip install torch transformers bitsandbytes accelerate datasets scikit-learn inspect-ai

# 2. Verify GPU status
nvidia-smi

# 3. Create the directories just in case (though Git should have them)
mkdir -p data/raw data/processed results/sweeps results/plots logs/calibration logs/inspect_traces notebooks

echo "--- Environment Ready. Ready for Llama-3-8B Sweep. ---"