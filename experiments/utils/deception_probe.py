import torch
import torch.nn as nn
import numpy as np

class OrthogonalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_direction = None
        # Expose centroids for visualization
        self.mu_HU = None
        self.mu_DU = None
        self.mu_HM = None

    def fit(self, acts_HU, acts_DU, acts_HM, verbose=True):
        def to_f32(t):
            if isinstance(t, np.ndarray): t = torch.tensor(t)
            return t.float()

        # Store centroids
        self.mu_HU = torch.mean(to_f32(acts_HU), dim=0)
        self.mu_DU = torch.mean(to_f32(acts_DU), dim=0)
        self.mu_HM = torch.mean(to_f32(acts_HM), dim=0)
        
        v_raw = self.mu_DU - self.mu_HU
        v_sit = self.mu_HM - self.mu_HU
        
        norm_sit_sq = torch.dot(v_sit, v_sit)
        
        if norm_sit_sq < 1e-9:
            self.final_direction = v_raw
        else:
            proj = (torch.dot(v_raw, v_sit) / norm_sit_sq) * v_sit
            self.final_direction = v_raw - proj
            
        self.final_direction = self.final_direction / (self.final_direction.norm() + 1e-8)

    def score(self, activations):
        if self.final_direction is None: raise ValueError("Probe not fitted.")
        if isinstance(activations, np.ndarray): activations = torch.tensor(activations)
        
        activations = activations.float()
        if self.final_direction.device != activations.device:
            self.final_direction = self.final_direction.to(activations.device)
            
        if activations.dim() == 1:
            return torch.dot(activations, self.final_direction).item()
        else:
            return torch.matmul(activations, self.final_direction).numpy()