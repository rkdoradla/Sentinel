# experiments/run_baseline.py
import torch
import numpy as np
from experiments.utils.deception_probe import DeceptionProbe
from experiments.utils.pareto import ParetoCombiner

def get_mock_data(num_samples=200, dim=4096, seed=42):
    """Generates mock activations for cheap (noisy) vs expensive (clean) probes."""
    torch.manual_seed(seed)
    labels = torch.randint(0, 2, (num_samples,))
    
    # Cheap Data: Noisy signal (harder to separate)
    acts_cheap = torch.randn(num_samples, dim) + (labels.unsqueeze(1) * 0.4)
    
    # Expensive Data: Clear signal (easier to separate)
    acts_exp = torch.randn(num_samples, dim) + (labels.unsqueeze(1) * 1.5)
    
    return acts_cheap, acts_exp, labels

def start_experiment():
    print("=== 1. Setup Phase ===")
    # Initialize TWO probes
    probe_cheap = DeceptionProbe()
    probe_exp = DeceptionProbe()

    # Generate Training Data and Validation Data
    # (Validation is needed to fit the Pareto thresholds without overfitting)
    train_c, train_e, train_y = get_mock_data(200, seed=42)
    val_c, val_e, val_y = get_mock_data(200, seed=43)

    print("=== 2. Training Phase ===")
    probe_cheap.train(train_c, train_y)
    probe_exp.train(train_e, train_y)

    print("=== 3. Pareto Optimization Phase ===")
    # Calculate scores on validation set
    # (We cast to numpy because ParetoCombiner uses scipy/numpy)
    scores_cheap = np.array([probe_cheap.calculate_score(x) for x in val_c])
    scores_exp   = np.array([probe_exp.calculate_score(x) for x in val_e])
    labels_np    = val_y.numpy()

    # Initialize Combiner
    # cost_cheap=1, cost_exp=10 -> Expensive is 10x slower/costlier
    combiner = ParetoCombiner(cost_cheap=1.0, cost_exp=10.0, cost_intervention=50.0)

    # Find the optimal 'Deferral Region' subject to a budget constraint (e.g., 15.0)
    combiner.fit_optimization(
        s1_val=scores_cheap, 
        s2_val=scores_exp, 
        labels=labels_np,
        budget_constraint=15.0 
    )

    print("=== 4. Inference Phase (Test) ===")
    # Create test vectors (representing the same input processed by different layers)
    test_vec_c = torch.randn(4096)
    test_vec_e = torch.randn(4096)

    # Define wrappers for Lazy Evaluation
    # The combiner calls these ONLY if needed
    def call_cheap(x):
        return probe_cheap.calculate_score(test_vec_c) # In real usage, pass 'x'

    def call_expensive(x):
        print("   >> [SYSTEM] Uncertainty detected! Triggering Expensive Monitor...")
        return probe_exp.calculate_score(test_vec_e)

    # Run Prediction
    decision = combiner.predict(
        x=None, # Passed to wrappers, unused here because we use closures for mock data
        monitor1_func=call_cheap,
        monitor2_func=call_expensive
    )

    status = "INTERVENE (Unsafe)" if decision == 1 else "PASS (Safe)"
    print(f"Final Decision: {status}")

if __name__ == "__main__":
    start_experiment()