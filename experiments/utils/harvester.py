import torch

class ActivationHarvester:
    """
    The 'Universal Harvester': Captures hidden activations from specific 
    layers for both Llama-based and GPT-2 architectures.
    """
    def __init__(self):
        self.activations = None

    def hook_fn(self, module, input, output):
        # Handles models that return tuples (like Llama/GPT-2) vs raw tensors
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def harvest(self, model, layer_idx, input_ids):
        # --- DYNAMIC LAYER FINDER ---
        # 1. Llama-3 Path
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            target_layer = model.model.layers[layer_idx]
        # 2. GPT-2 Path (for @As's baseline tests)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            target_layer = model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Unknown architecture: {type(model)}")

        # Register hook, run model, and clean up
        handle = target_layer.register_forward_hook(self.hook_fn)
        with torch.no_grad():
            # We ensure hidden states are calculated
            model(input_ids, output_hidden_states=True)
        handle.remove()
        
        return self.activations