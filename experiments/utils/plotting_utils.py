import matplotlib.pyplot as plt
import numpy as np
import os

def plot_deception_trajectory(thoughts, risk_scores, filename="deception_pivot.png"):
    """
    Visualizes the temporal 'Pivot to Deception' across model monologues.
    """
    # Use 'Agg' to ensure it works on cloud servers without monitors
    import matplotlib
    matplotlib.use('Agg')
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    indices = np.arange(len(thoughts))

    # Plot the Risk Curve
    ax.plot(indices, risk_scores, marker='o', linestyle='-', color='#e74c3c', linewidth=2, label='Deception Risk')
    ax.fill_between(indices, risk_scores, color='#e74c3c', alpha=0.1)

    # Add a Safety Threshold Line
    ax.axhline(y=0.5, color='#2ecc71', linestyle='--', label='Safety Threshold')

    # Formatting
    ax.set_title("Sentinel: Temporal Intent Monitoring", fontsize=14, fontweight='bold')
    ax.set_xlabel("Reasoning Step (Chronological)", fontsize=12)
    ax.set_ylabel("Probability of Deceptive Intent", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"Step {i+1}" for i in indices], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save logic (using absolute path to root_dir/results)
    # This matches the os.makedirs in your main.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    save_path = os.path.join(root_dir, "results", filename)
    
    plt.savefig(save_path)
    plt.close()