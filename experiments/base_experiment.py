import torch
from .quantization_config import get_4bit_config

"""
Sentinel Base Experiment Template
Architected by: Venkatesh Doradla (@rkdoradla)
Purpose: Standardizing the 'Precision-Safety Gap' sweep for Colab-to-GitHub porting.
"""

class SentinelExperiment:
    def __init__(self, model_id, precision="nf4"):
        self.model_id = model_id
        self.config = get_4bit_config() if precision == "nf4" else None
        
    def run_inference(self, prompt):
        """Standardized wrapper for model calls to ensure quantization is active."""
        # This is where As/Bryan will drop their specific Colab logic
        pass

    def log_metrics(self, g_mean_score):
        """Standardized logging for Bryan's Pareto Frontier analysis."""
        print(f"Model: {self.model_id} | G-Mean: {g_mean_score}")
