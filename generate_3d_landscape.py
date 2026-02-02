import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

def generate_3d_basin():
    # 1. Load the finalized metrics
    try:
        with open("results/final_sweep_metrics.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: results/final_sweep_metrics.json not found!")
        return

    # 2. Extract Data for Plotting
    # We use log-scale for sigma to better visualize the 'cliff'
    sigmas = np.array([r['sigma'] for r in data["Control (L5)"]])
    l5_drift = np.array([r['drift'] for r in data["Control (L5)"]])
    l31_drift = np.array([r['drift'] for r in data["Target (L31)"]])

    # 3. Interpolation for a Smooth Surface
    layers = np.array([5, 31])
    sigma_grid = np.logspace(np.log10(sigmas.min()), np.log10(sigmas.max()), 50)
    layer_grid = np.linspace(0, 32, 50)
    
    drift_data = np.vstack([l5_drift, l31_drift])
    f_interp = interp2d(sigmas, layers, drift_data, kind='linear')
    Z = f_interp(sigma_grid, layer_grid)

    # 4. Generate 3D Figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.log10(sigma_grid), layer_grid)
    
    # Use 'magma' to highlight the heat/energy of the deceptive manifold
    surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', alpha=0.9)
    
    # Vertical Line at NF4 Threshold (6% noise)
    nf4_log = np.log10(0.06)
    ax.plot([nf4_log, nf4_log], [0, 32], [0, Z.max()], color='cyan', linestyle='--', linewidth=3, label="NF4 Horizon")

    # 5. Labels & Formatting
    ax.set_title("Figure 5: Topological Deception Landscape (Llama-3-8B)", fontsize=16)
    ax.set_xlabel("Noise Intensity (log10 Sigma)")
    ax.set_ylabel("Model Layer")
    ax.set_zlabel("Representational Drift")
    ax.view_init(elev=35, azim=225) # Optimized angle to see the 'Cliff' at L31
    
    plt.colorbar(surf, shrink=0.5, label="Instability")
    plt.savefig("results/Figure_5_3D_Sharp_Basin.png", dpi=300)
    print("✔ Visual Breakthrough Generated: results/Figure_5_3D_Sharp_Basin.png")

if __name__ == "__main__":
    generate_3d_basin()
