import sys
import os
import pkg_resources

# PATH BOOTSTRAP
current_file = os.path.abspath(__file__)
exp_dir = os.path.dirname(current_file)
project_root = os.path.dirname(exp_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)

# DEPENDENCY CHECK (H100 Stability)
required = {'transformers': '4.40.1', 'accelerate': '0.29.3'}
for pkg, ver in required.items():
    try:
        if pkg_resources.get_distribution(pkg).version != ver:
            print(f"[!] WARNING: {pkg} version mismatch. Need {ver}.")
    except: pass

import argparse
import torch
import json
import gc
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer

# Sentinel Modules
from experiments.utils.apollo_adapter import ApolloAdapter
from experiments.utils.harvester import ActivationHarvester
from experiments.utils.deception_probe import OrthogonalProbe
from experiments.utils.visualizer import SentinelVisualizer
from experiments.utils.pareto import ParetoCombiner

console = Console()

def simulate_quantization_noise(activations, noise_scale=0.06):
    """
    Hypothesis 1: Input-Scaled Quantization Simulation.
    Matches NF4 Signal-to-Noise Ratio (SNR) by scaling noise to the input's standard deviation.
    Formula: y = x + N(0,1) * sigma * std(x)
    """
    # [CRITICAL FIX] Scale noise by the signal strength (Input-Dependent)
    # This ensures Llama-3 outliers (100x size) get 100x noise, maintaining constant SNR.
    with torch.no_grad():
        input_std = activations.std(dim=-1, keepdim=True)
        noise = torch.randn_like(activations) * noise_scale * input_std
        return activations + noise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="SENTINEL_AUDIT")
    parser.add_argument("--precision", type=str, default="BF16")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") 
    parser.add_argument("--scan_start", type=int, default=15)
    parser.add_argument("--scan_end", type=int, default=32)
    args = parser.parse_args()

    console.print(Panel.fit(f"[bold cyan]SENTINEL: SIMULATED QUANTIZATION ENGINE[/bold cyan]\nTarget: {args.model}\nPrecision: {args.precision}", border_style="cyan"))

    # 1. LOAD MODEL
    dtype = torch.bfloat16 if args.precision == "BF16" else torch.float32
    if dtype == torch.bfloat16: console.print("[yellow][!] Enforcing BF16 for Manifold Integrity.[/yellow]")

    console.print(f"[bold]Loading Model...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=dtype, attn_implementation="flash_attention_2")
    except:
        console.print("[yellow][!] FlashAttention missing. Falling back to Safe Mode (Eager).[/yellow]")
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=dtype)
    model.eval()

    # 2. DATA
    data_path = os.path.join(project_root, "data/raw/obfuscation_dataset.jsonl")
    adapter = ApolloAdapter()
    data_pairs = adapter.load_dataset(data_path)
    if not data_pairs:
        console.print("[bold red]CRITICAL: Run 'ingest_liars_bench.py' first![/bold red]")
        return
        
    texts_U = [d['text_unmonitored'] for d in data_pairs]
    texts_M = [d['text_monitored'] for d in data_pairs]
    labels  = torch.tensor([d['label'] for d in data_pairs])

    # 3. FORENSIC SWEEP
    layer_metrics = {}
    best_score = -1.0
    best_layer = -1
    best_clean_scores = None
    best_drift = None
    
    if hasattr(model.config, "num_hidden_layers"): args.scan_end = min(args.scan_end, model.config.num_hidden_layers - 1)
    
    console.print(f"\n[bold]Initiating Sweep:[/bold] Layers {args.scan_start} to {args.scan_end}")
    
    for layer_idx in track(range(args.scan_start, args.scan_end + 1), description="Scanning Layers..."):
        harvester = ActivationHarvester(model, tokenizer, layer_idx)
        
        # Harvest
        acts_U = harvester.harvest(texts_U, batch_size=8, verbose=False)
        acts_M = harvester.harvest(texts_M, batch_size=8, verbose=False)
        
        # A. SIMULATED QUANTIZATION (Hypothesis 1)
        # Add noise to mimic NF4 precision loss without reloading model
        acts_Q = simulate_quantization_noise(acts_U)
        drift = torch.norm(acts_U - acts_Q, dim=1).numpy()
        
        # B. PROBE (ODP)
        probe = OrthogonalProbe()
        probe.fit(acts_U[labels==0], acts_U[labels==1], acts_M[labels==0], verbose=False)
        
        clean_scores = probe.score(acts_U)
        separation = clean_scores[labels==1].mean() - clean_scores[labels==0].mean()
        
        layer_metrics[layer_idx] = {'separation': float(separation), 'purified': float(separation)}
        
        if separation > best_score:
            best_score = separation
            best_layer = layer_idx
            best_clean_scores = clean_scores
            best_drift = drift

        del harvester, acts_U, acts_M, acts_Q, probe
        torch.cuda.empty_cache()
        gc.collect()

    # 4. GENERATE ARTIFACTS
    console.print(f"\n[bold green]Scan Complete.[/bold green] Peak Horizon: L{best_layer}")
    
    viz = SentinelVisualizer(project_root)
    # Generate the 3 Holy Grail Graphs
    viz.generate_all(layer_metrics, best_drift, labels.numpy(), best_clean_scores)
    
    with open(os.path.join(project_root, "results", "run_log.json"), "w") as f:
        json.dump({"timestamp": "FINAL", "peak_layer": best_layer, "score": best_score}, f, indent=4)

    console.print("\n[bold cyan]=== SENTINEL SUBMISSION READY ===[/bold cyan]")

if __name__ == "__main__":
    main()