# experiments/run_baseline.py
import torch
from experiments.utils.deception_probe import DeceptionProbe

def start_experiment():
    # 1. Initialize the probe you just wrote
    probe = DeceptionProbe()
    
    # 2. Mock data for 'Beginner Mode' testing
    # In reality, these are the 'latents' from Layer 22
    mock_activations = torch.randn(10, 4096) # 10 samples, Llama size
    mock_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # 3. Train it!
    print("Starting training on local CPU...")
    probe.train(mock_activations, mock_labels)

    # 4. Test a single 'thought'
    test_vector = torch.randn(4096)
    score = probe.calculate_score(test_vector)
    print(f"Deception Score: {score:.4f}")

if __name__ == "__main__":
    start_experiment()