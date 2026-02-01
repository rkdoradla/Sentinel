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
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer_idx]
        return self.model.transformer.h[self.layer_idx]

    def _hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # Immediate CPU Offload
        self.activations.append(h[:, -1, :].detach().to(dtype=torch.float32).cpu())

    def harvest(self, texts, batch_size=8, verbose=True):
        self.activations = []
        layer = self._get_layer_module()
        self.hook_handle = layer.register_forward_hook(self._hook_fn)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            it = range(0, len(texts), batch_size)
            if verbose: it = tqdm(it, desc="Harvesting", leave=False)
            with torch.no_grad():
                for i in it:
                    batch = texts[i:i+batch_size]
                    if not batch: continue
                    inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
                    self.model(**inputs)
                    del inputs
        finally:
            self.hook_handle.remove()
        return torch.cat(self.activations, dim=0) if self.activations else torch.empty(0)