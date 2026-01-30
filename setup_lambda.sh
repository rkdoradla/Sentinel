# Day 1 Setup Script for Sentinel
# Run this the moment you log into Lambda!

echo "--- Initializing Sentinel Research Environment ---"

# 1. Install necessary AI libraries
pip install torch transformers bitsandbytes accelerate datasets

# 2. Verify GPU status
nvidia-smi

# 3. Create the directories just in case (though Git should have them)
mkdir -p data/raw results/sweeps logs/calibration

echo "--- Environment Ready. Ready for Llama-3-8B Sweep. ---"