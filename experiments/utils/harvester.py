import torch
from tqdm import tqdm

class ActivationHarvester:
    """
    Extracts hidden states from a target layer.
    Universal Logic: Works for GPT-2, Llama-2, Llama-3, Mistral, etc.
    """
    def __init__(self, model, tokenizer, layer_idx):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = model.device
        self.activations = []
        self.hook_handle = None

    def _get_layer_module(self):
        """
        Identifies the correct PyTorch module for the specific layer index
        based on the model architecture.
        """
        # Llama / Mistral / Qwen Architecture
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            if self.layer_idx >= len(self.model.model.layers):
                raise ValueError(f"Layer {self.layer_idx} out of bounds for Llama model.")
            return self.model.model.layers[self.layer_idx]
        
        # GPT-2 Architecture
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            if self.layer_idx >= len(self.model.transformer.h):
                raise ValueError(f"Layer {self.layer_idx} out of bounds for GPT-2.")
            return self.model.transformer.h[self.layer_idx]
            
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")

    def _hook_fn(self, module, input, output):
        """
        The hook function that runs during the forward pass.
        Captures the hidden state of the LAST token.
        """
        # output is usually a tuple (hidden_state, past_key_values)
        if isinstance(output, tuple):
            hidden_state = output[0]
        else:
            hidden_state = output
        
        # We take the vector corresponding to the last token in the sequence
        # Shape: [Batch_Size, Sequence_Length, Hidden_Dim] -> [Batch_Size, Hidden_Dim]
        # detach() is crucial to prevent memory leaks
        last_token_act = hidden_state[:, -1, :].detach().cpu()
        self.activations.append(last_token_act)

    def harvest(self, texts, batch_size=8):
        """
        Runs inference on the provided texts and extracts activations.
        """
        self.activations = []
        target_layer = self._get_layer_module()
        
        # Register the hook
        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)
        
        print(f"[Harvester] Extracting vectors from Layer {self.layer_idx} on {self.device}...")
        
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            # Process in batches
            for i in tqdm(range(0, len(texts), batch_size), desc="Harvesting"):
                batch_texts = texts[i : i + batch_size]
                if not batch_texts: continue
                
                # Tokenize
                # Note: Left-padding (handled in main.py) is crucial for correct index extraction
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                
                with torch.no_grad():
                    self.model(**inputs)
        
        finally:
            # Always clean up hooks
            if self.hook_handle:
                self.hook_handle.remove()
        
        if not self.activations:
            return torch.empty(0)
            
        # Concatenate all batches
        return torch.cat(self.activations, dim=0)