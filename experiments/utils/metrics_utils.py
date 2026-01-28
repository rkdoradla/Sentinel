import numpy as np

"""
Sentinel Framework: Metrics & Evaluation Utility
Lead Author: Venkatesh Doradla (@rkdoradla)
Primary Functional Role: Infrastructure & Research Framework Lead
Purpose: Implementation of g-mean² for balancing Safety Trilemma trade-offs.
"""

def calculate_g_mean_sq(tpr: float, tnr: float) -> float:
    """
    Calculates the Geometric Mean Squared to balance sensitivity (TPR) and specificity (TNR).
    This serves as the primary Y-axis metric for our Pareto Frontier analysis.
    
    Formula: g-mean^2 = TPR * TNR
    Reference: Sentinel Project Technical Strategy
    """
    return float(tpr * tnr)

def detect_safety_collapse(current_score: float, baseline_score: float, threshold: float = 0.15) -> bool:
    """
    Identifies if a model has fallen into the 'Precision-Safety Gap'.
    Returns True if safety metrics drop significantly below the FP16 baseline 
    due to quantization-induced noise.
    """
    drop = baseline_score - current_score
    return drop > threshold

def format_trilemma_payload(efficiency: float, assurance: float, observability: float) -> dict:
    """
    Standardizes the data payload for the Final Pareto Frontier visualization.
    E: Inference Efficiency (Quantization Level)
    A: Adversarial Assurance (g-mean² Score)
    O: Observability of Intent (Latent Extraction Quality)
    """
    return {
        "efficiency_score": efficiency,
        "assurance_score": assurance,
        "observability_score": observability
    }
