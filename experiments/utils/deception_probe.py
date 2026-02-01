import torch
import torch.nn as nn
import numpy as np

class OrthogonalProbe(nn.Module):
    """
    Implements the 'Orthogonal Subspace Projection' for Sentinel.
    Math: v_purified = v_raw - Proj_v_sit(v_raw)
    """
    def __init__(self):
        super().__init__()
        self.final_direction = None

    def fit(self, acts_HU, acts_DU, acts_HM, verbose=True):
        """
        acts_HU: Honest, Unmonitored (Baseline)
        acts_DU: Deceptive, Unmonitored (Raw Intent)
        acts_HM: Honest, Monitored (Situational Awareness)
        """
        # Convert to tensor if needed and ensure float32 for geometric stability
        def to_f32(t):
            if isinstance(t, np.ndarray): t = torch.tensor(t)
            return t.float()

        h_u = to_f32(acts_HU)
        d_u = to_f32(acts_DU)
        h_m = to_f32(acts_HM)

        # 1. Centroids
        mu_HU = torch.mean(h_u, dim=0)
        mu_DU = torch.mean(d_u, dim=0)
        mu_HM = torch.mean(h_m, dim=0)
        
        # 2. Vector Extraction
        v_raw = mu_DU - mu_HU        # Raw Intent (Contaminated)
        v_sit = mu_HM - mu_HU        # Situational Awareness
        
        # 3. Gram-Schmidt Purification
        norm_sit_sq = torch.dot(v_sit, v_sit)
        
        if norm_sit_sq < 1e-9:
            self.final_direction = v_raw
        else:
            # Projection of Intent onto Situation
            proj = (torch.dot(v_raw, v_sit) / norm_sit_sq) * v_sit
            # Subtract projection to get Orthogonal component
            self.final_direction = v_raw - proj
            
        # Normalize
        self.final_direction = self.final_direction / (self.final_direction.norm() + 1e-8)
        
        if verbose:
            print(f"[Probe] Purified Norm: {self.final_direction.norm():.4f}")

    def score(self, activations):
        if self.final_direction is None: raise ValueError("Probe not fitted.")
        
        if isinstance(activations, np.ndarray):
            activations = torch.tensor(activations, dtype=torch.float32)
            
        # Ensure device match
        if self.final_direction.device != activations.device:
            self.final_direction = self.final_direction.to(activations.device)
            
        if activations.dim() == 1:
            return torch.dot(activations, self.final_direction).item()
        else:
            return torch.matmul(activations, self.final_direction).numpy()