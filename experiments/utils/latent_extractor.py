import torch
import torch.nn as nn

class OrthogonalLatentExtractor:
    """
    Implements the 'Gram-Schmidt' Orthogonal Intent Projection logic.
    Designed to isolate true deceptive intent from situational awareness noise.
    """
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.theta_clean = None

    def calibrate_vectors(self, activations, labels, is_monitored):
        """
        Derives the 'Clean Intent' vector using the 2x2 Matrix Logic.
        
        Args:
            activations: (N, d_model) tensor of hidden states
            labels: (N,) tensor where 1 = Deceptive, 0 = Honest
            is_monitored: (N,) tensor where 1 = Monitored Context, 0 = Unmonitored
        """
        # 1. Define masks for the 2x2 matrix
        mask_HU = (labels == 0) & (is_monitored == 0) # Honest/Unmonitored
        mask_HM = (labels == 0) & (is_monitored == 1) # Honest/Monitored
        mask_DU = (labels == 1) & (is_monitored == 0) # Deceptive/Unmonitored

        # 2. Safety Check: Ensure we have data for all centroids to avoid NaN/Zero
        if mask_HU.sum() == 0 or mask_HM.sum() == 0 or mask_DU.sum() == 0:
            print(f"Warning: Missing data groups in Layer {self.layer_idx}. Skipping calibration.")
            return None

        # 3. Calculate Mean Vectors (Centroids)
        mu_HU = activations[mask_HU].mean(dim=0)
        mu_HM = activations[mask_HM].mean(dim=0)
        mu_DU = activations[mask_DU].mean(dim=0)

        # 4. Calculate Raw Directions (Deltas)
        theta_situation = mu_HM - mu_HU  # The "Being Watched" signal
        theta_intent_raw = mu_DU - mu_HU # The "Deceiving" signal (polluted)

        # 5. Gram-Schmidt Orthogonalization (Rejection)
        # Epsilon (1e-8) prevents divide-by-zero if vectors collapse in NF4
        dot_prod = torch.dot(theta_intent_raw, theta_situation)
        norm_sq = torch.dot(theta_situation, theta_situation) + 1e-8
        
        # Calculate and subtract the projection
        projection = (dot_prod / norm_sq) * theta_situation
        self.theta_clean = theta_intent_raw - projection
        
        return self.theta_clean

    def get_intent_score(self, live_activations):
        """
        Projects live hidden states onto the normalized clean intent vector.
        """
        if self.theta_clean is None:
            # Safe fallback if calibration hasn't run
            return torch.zeros(live_activations.shape[0], device=live_activations.device)
        
        # Normalize the vector to ensure consistent scoring across FP16 and NF4
        u_clean = self.theta_clean / (torch.norm(self.theta_clean) + 1e-8)
        
        # Linear projection score
        return torch.matmul(live_activations, u_clean)
