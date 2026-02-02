import torch
from tqdm import tqdm

class ActivationHarvester:
    def __init__(self, model, tokenizer, layer_idx):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = model.device
        self.activations = []
        self.hook_handle = None

    def _get_layer_module(self):
        # Support for Llama, Mistral, and GPT-style architectures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[self.layer_idx]
        else:
            raise ValueError("Unknown architecture")

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple): 
            hidden_state = output[0]
        else: 
            hidden_state = output
        
        # OFF-LOAD TO RAM: Detach and move to CPU immediately
        # We cast to float32 to ensure mathematical stability during later analysis
        last_token = hidden_state[:, -1, :].detach().to(dtype=torch.float32).cpu()
        self.activations.append(last_token)

    def harvest(self, texts, batch_size=8, verbose=True):
        self.activations = []
        target_layer = self._get_layer_module()
        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)
        
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        try:
            range_gen = range(0, len(texts), batch_size)
            if verbose: 
                range_gen = tqdm(range_gen, desc=f"Harvesting Layer {self.layer_idx}")
            
            with torch.no_grad():
                for i in range_gen:
                    batch_texts = texts[i : i + batch_size]
                    if not batch_texts: continue
                    inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
                    self.model(**inputs)
                    del inputs
        finally:
            if self.hook_handle: 
                self.hook_handle.remove()
                
        if not self.activations: 
            return torch.empty(0)
        return torch.cat(self.activations, dim=0)
