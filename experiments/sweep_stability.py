import torch, json, os, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.utils.apollo_adapter import ApolloAdapter

# 1. FIX: Custom Encoder for NumPy types (Prevents JSON TypeError)
class SentinelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return super().default(obj)

# 2. FIX: Transparent Hook (Prevents IndexError)
class NoiseInjectionHook:
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        std = hidden_states.std(dim=-1, keepdim=True)
        noise = torch.randn_like(hidden_states) * self.sigma * std
        modified = hidden_states + noise
        if is_tuple: return (modified,) + output[1:] # Preserve the KV-cache tuple
        return modified

def run_final_phase2():
    layers_to_test = {"Target (L31)": 31, "Control (L5)": 5}
    sigmas = [0.001, 0.005, 0.01, 0.05, 0.06, 0.1, 0.2]
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    print(f"🚀 Starting Final Phase 2 Audit...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    adapter = ApolloAdapter()
    data = adapter.load_dataset("data/raw/obfuscation_dataset.jsonl")
    texts = [d['text_unmonitored'] for d in data[:25]] # Safe batch size
    
    all_results = {}
    os.makedirs("results", exist_ok=True)

    for label, idx in layers_to_test.items():
        print(f"\n🔍 Scanning {label}...")
        layer_results = []
        target_module = model.model.layers[idx]
        
        for s in sigmas:
            print(f"   Sigma {s}...", end=" ", flush=True)
            hook_handle = target_module.register_forward_hook(NoiseInjectionHook(s))
            total_drift = 0
            with torch.no_grad():
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                    outputs = model(**inputs)
                    total_drift += torch.norm(outputs.logits).item()
                    del inputs, outputs
            hook_handle.remove()
            layer_results.append({"sigma": s, "drift": total_drift / len(texts)})
            print("Done.")
        all_results[label] = layer_results

    # 3. SAVE DATA
    with open("results/final_sweep_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4, cls=SentinelEncoder)

    # 4. VISUALIZATION (Integrated)
    print("\n📊 Generating Figure 4...")
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    for label, results in all_results.items():
        plt.plot([r['sigma'] for r in results], [r['drift'] for r in results], marker='o', label=label, linewidth=2)
    
    plt.xscale('log')
    plt.axvline(x=0.06, color='gray', linestyle='--', label="NF4 Horizon (6%)")
    plt.title("Figure 4: Stability Plot - Target vs. Control", fontsize=14)
    plt.xlabel("Perturbation Sigma (Log Scale)")
    plt.ylabel("Output Manifold Drift")
    plt.legend()
    plt.savefig("results/Figure_4_Comparative_Stability.png")
    
    print("\n✔ Phase 2 Comparative Sweep Complete.")
    print("✔ Figure 4 saved to results/")

if __name__ == "__main__":
    run_final_phase2()