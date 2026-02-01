import torch
import torch.nn as nn
import numpy as np

class OrthogonalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_direction = None

    def fit(self, acts_HU, acts_DU, acts_HM, verbose=True):
        def to_f32(t): return torch.tensor(t).float() if isinstance(t, np.ndarray) else t.float()
        
        mu_HU = torch.mean(to_f32(acts_HU), dim=0)
        mu_DU = torch.mean(to_f32(acts_DU), dim=0)
        mu_HM = torch.mean(to_f32(acts_HM), dim=0)
        
        v_raw = mu_DU - mu_HU
        v_sit = mu_HM - mu_HU
        
        norm_sit_sq = torch.dot(v_sit, v_sit)
        if norm_sit_sq < 1e-9: self.final_direction = v_raw
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
        if activations.dim() == 1: return torch.dot(activations, self.final_direction).item()
        return torch.matmul(activations, self.final_direction).numpy()
    
    # Alias for visualizer compatibility
    def score_purified(self, activations): return self.score(activations)