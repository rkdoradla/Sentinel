import torch, json, os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.utils.apollo_adapter import ApolloAdapter

class NoiseInjectionHook:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, module, input, output):
        # Determine if output is a tuple (Standard for Llama)
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        # ACCURATE MATH: sigma * std(x)
        std = hidden_states.std(dim=-1, keepdim=True)
        noise = torch.randn_like(hidden_states) * self.sigma * std
        modified = hidden_states + noise
        
        # FIX: Return the modified state while preserving the rest of the tuple
        if is_tuple:
            # We recreate the tuple with the modified hidden state as the first element
            return (modified,) + output[1:]
        return modified

def run_comparative_sweep():
    layers_to_test = {"Target (L31)": 31, "Control (L5)": 5}
    sigmas = [0.001, 0.005, 0.01, 0.05, 0.06, 0.1, 0.2] # Sweep covers the agreed Sigmas
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    print(f"🚀 Executing Comparative Phase 2 Sweep...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    adapter = ApolloAdapter()
    data = adapter.load_dataset("data/raw/obfuscation_dataset.jsonl")
    texts = [d['text_unmonitored'] for d in data[:50]]
    
    all_results = {}

    for label, idx in layers_to_test.items():
        print(f"Scanning {label}...")
        layer_results = []
        target_module = model.model.layers[idx]
        
        for s in sigmas:
            hook_handle = target_module.register_forward_hook(NoiseInjectionHook(s))
            total_drift = 0
            with torch.no_grad():
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                    outputs = model(**inputs)
                    # Measuring Logit Drift as a proxy for representational collapse
                    total_drift += torch.norm(outputs.logits).item()
            
            hook_handle.remove()
            layer_results.append({"sigma": s, "drift": total_drift / len(texts)})
        
        all_results[label] = layer_results

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    for label, results in all_results.items():
        x = [r['sigma'] for r in results]
        y = [r['drift'] for r in results]
        plt.plot(x, y, marker='o', label=label, linewidth=2)

    plt.xscale('log')
    plt.axvline(x=0.06, color='gray', linestyle='--', label="NF4 Equivalent (6%)")
    plt.title("Figure 4: Stability Plot - Target vs. Control", fontsize=14)
    plt.xlabel("Perturbation Sigma (Log Scale)", fontsize=12)
    plt.ylabel("Output Manifold Drift", fontsize=12)
    plt.legend()
    plt.savefig("results/Figure_4_Comparative_Stability.png")
    
    # Save using the standard JSON format (ensure SentinelEncoder is used if needed)
    with open("results/final_sweep_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    run_comparative_sweep()