import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

class DeceptionProbe:
    def __init__(self):
        # CPU/Numpy parameters (for saving/loading)
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        
        # GPU/Torch parameters (cached for fast inference)
        self.torch_weights = None
        self.torch_bias = None
        self.torch_mean = None
        self.torch_std = None

    def _ensure_numpy(self, x):
        """
        Sanitizer: Safely converts CUDA tensors to CPU NumPy arrays 
        to prevent sklearn crashes.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.array(x)

    def train(self, activations, labels):
        """
        activations: (n_samples, d_model) - can be CUDA Tensor or Numpy
        labels: binary labels
        """
        # 1. Sanitize Inputs (Fixes the TypeError)
        X = self._ensure_numpy(activations)
        y = self._ensure_numpy(labels)

        # 2. Compute Normalization Statistics
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8

        # 3. Normalize Activations (Z-score)
        X_normalized = (X - self.mean) / self.std

        # 4. Fit Logistic Regression
        clf = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=1000)
        clf.fit(X_normalized, y)

        self.weights = clf.coef_[0]
        self.bias = clf.intercept_[0]
        
        # 5. Cache Torch Tensors for Inference
        # We create them here but move them to the correct device dynamically later
        self.torch_mean = torch.tensor(self.mean, dtype=torch.float32)
        self.torch_std = torch.tensor(self.std, dtype=torch.float32)
        self.torch_weights = torch.tensor(self.weights, dtype=torch.float32)
        self.torch_bias = torch.tensor(self.bias, dtype=torch.float32)
        
        print(f"Baseline Probe Trained. Training Accuracy: {clf.score(X_normalized, y):.2%}")

    def calculate_score(self, activation_vector):
        """
        Hybrid Inference:
        - If input is CUDA Tensor -> Runs purely on GPU (Fast)
        - If input is Numpy/List  -> Runs on CPU (Legacy)
        """
        if isinstance(activation_vector, torch.Tensor):
            return self._calculate_score_torch(activation_vector)
        
        # --- CPU Fallback ---
        x = self._ensure_numpy(activation_vector)
        
        # Normalize
        z = (x - self.mean) / self.std
        
        # FIX: Use @ operator (Matrix Multiplication) with input 'z' FIRST.
        # This handles both:
        #   1. Single Vector: (4096,) @ (4096,) -> Scalar
        #   2. Batch Matrix:  (N, 4096) @ (4096,) -> (N,) Vector
        logit = z @ self.weights + self.bias
        
        return 1 / (1 + np.exp(-logit))

    def _calculate_score_torch(self, x):
        """
        PyTorch-native inference. 
        Critical for running inside the model generation loop without cpu-sync overhead.
        """
        device = x.device
        
        # Lazy load weights to the correct device
        if self.torch_mean.device != device:
            self.torch_mean = self.torch_mean.to(device)
            self.torch_std = self.torch_std.to(device)
            self.torch_weights = self.torch_weights.to(device)
            self.torch_bias = self.torch_bias.to(device)

        # Normalize
        z = (x - self.torch_mean) / self.torch_std
        
        # Project (Linear Layer)
        # Handle both single vector (d_model) and batch (batch, d_model)
        if x.ndim > 1:
            logit = torch.matmul(z, self.torch_weights) + self.torch_bias
        else:
            logit = torch.dot(z, self.torch_weights) + self.torch_bias
            
        score = torch.sigmoid(logit)
        
        # Return float if scalar (compatibility), else return tensor
        return score.item() if score.numel() == 1 else score