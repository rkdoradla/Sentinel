import torch
from typing import Dict, List

"""
Sentinel Framework: Latent Representation Extractor
Lead Author: Venkatesh Doradla (@rkdoradla)
Primary Functional Role: Infrastructure & Interpretability Lead
Purpose: Surgical extraction of internal model states (Layer 22) to monitor intent.
"""

class LatentExtractor:
    def __init__(self, model, tokenizer):
        """
        Initializes the extractor with a loaded model and tokenizer.
        Works with Llama, Mistral, and Gemma families.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.latents = {}

    def _get_hook(self, name: str):
        """Internal hook function to capture the activations of a specific layer."""
        def hook(model, input, output):
            # Capture the last token's hidden state (represents the model's 'summary' thought)
            # output is typically a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.latents[name] = hidden_states[:, -1, :].detach().cpu()
        return hook

    def extract_layer_data(self, text: str, layer_idx: int = 22) -> torch.Tensor:
        """
        Extracts the latent vector from a specific layer (default: 22).
        This provides the 'raw thought' signal needed for the Precision-Safety Gap analysis.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Llama-3 layer naming convention: model.layers[idx]
        target_layer = self.model.model.layers[layer_idx]
        
        handle = target_layer.register_forward_hook(self._get_hook(f"layer_{layer_idx}"))
        
        with torch.no_grad():
            self.model(**inputs)
        
        handle.remove()  # Critical for memory management
        return self.latents[f"layer_{layer_idx}"]
