import torch
import torch.nn as nn
import numpy as np

class OrthogonalProbe(nn.Module):
    """
    Implements the 'Orthogonal Projection' Strategy (Sentinel Architecture).
    
    Problem:
        A standard probe confuses 'Awareness of Monitoring' with 'Deception'.
    
    Solution:
        We calculate two vectors:
        1. v_intent: Difference between Deceptive and Honest behavior (Unmonitored)
        2. v_situation: Difference between Monitored and Unmonitored (Honest behavior)
        
        We then project v_situation OUT of v_intent via Gram-Schmidt.
    """
    def __init__(self):
        super().__init__()
        self.final_direction = None
        self.device = torch.device('cpu')

    def fit(self, acts_HU, acts_DU, acts_HM):
        """
        Fits the probe using 3 quadrants of the 2x2 matrix:
        
        acts_HU: Honest, Unmonitored (Baseline)
        acts_DU: Deceptive, Unmonitored (Intent Signal)
        acts_HM: Honest, Monitored (Situation Signal)
        """
        # Ensure inputs are tensors and float32
        if isinstance(acts_HU, np.ndarray): acts_HU = torch.tensor(acts_HU)
        if isinstance(acts_DU, np.ndarray): acts_DU = torch.tensor(acts_DU)
        if isinstance(acts_HM, np.ndarray): acts_HM = torch.tensor(acts_HM)

        # 1. Calculate Mean Vectors (Centroids)
        mu_HU = torch.mean(acts_HU.float(), dim=0)
        mu_DU = torch.mean(acts_DU.float(), dim=0)
        mu_HM = torch.mean(acts_HM.float(), dim=0)
        
        # 2. Derive Raw Directions
        # Vector representing pure deceptive intent (in the absence of monitoring)
        theta_intent = mu_DU - mu_HU
        
        # Vector representing the model noticing the system prompt
        theta_situation = mu_HM - mu_HU
        
        # 3. Gram-Schmidt Orthogonalization
        # We want a direction 'v' that aligns with intent but is orthogonal to situation.
        # v_clean = v_intent - proj_{situation}(v_intent)
        
        u = theta_situation
        v = theta_intent
        
        norm_u_sq = torch.dot(u, u)
        
        if norm_u_sq < 1e-9:
            # If situational vector is zero (model didn't notice prompt), just use intent
            print("[OrthogonalProbe] Warning: Situational vector is near zero. Using raw intent.")
            self.final_direction = v
        else:
            # Projection formula: (v . u / u . u) * u
            proj = (torch.dot(v, u) / norm_u_sq) * u
            self.final_direction = v - proj
            
        # Normalize the final direction
        self.final_direction = self.final_direction / (self.final_direction.norm() + 1e-8)
        
        print(f"[OrthogonalProbe] Projection complete. Monitoring signal removed.")
        print(f"   Original Intent Norm: {theta_intent.norm():.4f}")
        print(f"   Cleaned Direction Norm: {self.final_direction.norm():.4f}")

    def calculate_score(self, activations):
        """
        Returns scalar scores: Dot product with clean direction.
        High Score = Deceptive Intent
        """
        if self.final_direction is None:
            raise ValueError("Probe not fitted! Run .fit() first.")
        
        if isinstance(activations, np.ndarray):
            activations = torch.tensor(activations, dtype=torch.float32)
            
        # Move direction to same device as input
        self.final_direction = self.final_direction.to(activations.device)
            
        if activations.dim() == 1:
            return torch.dot(activations, self.final_direction).item()
        else:
            return torch.matmul(activations, self.final_direction).numpy()

    # Alias for compatibility
    def score(self, activations):
        return self.calculate_score(activations)