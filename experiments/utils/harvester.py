import torch

class ActivationHarvester:
    """
    The 'Harvester': Captures hidden activations from specific LLM layers 
    to feed into the DeceptionProbe.
    """
    def __init__(self):
        self.activations = None

    def hook_fn(self, module, input, output):
        # FIX: Llama-3 returns a tuple (hidden_states, cache, etc.)
        # We only want the first element [0], which is the hidden state tensor.
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def harvest(self, model, layer_idx, input_ids):
        """
        Runs a forward pass and catches the 'thought vector' at the specific layer.
        """
        # Target the specific layer (Layer 22 is the 'Lie Detector' spot for Llama-3-70B)
        # Note: This path 'model.model.layers' assumes a HuggingFace Llama architecture.
        try:
            target_layer = model.model.layers[layer_idx]
        except AttributeError:
            # Fallback for some quantized/PEFT wrapped models
            target_layer = model.layers[layer_idx]
        
        # 1. Register the 'hook' (The Wiretap)
        handle = target_layer.register_forward_hook(self.hook_fn)
        
        # 2. Run the model forward pass (Silent run, no generation)
        with torch.no_grad():
            model(input_ids, output_hidden_states=True)
        
        # 3. Remove the hook immediately (Cleanup)
        handle.remove()
        
        return self.activations