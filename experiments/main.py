import argparse
import torch
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer

# Sentinel Modules
from experiments.utils.apollo_adapter import ApolloAdapter
from experiments.utils.harvester import ActivationHarvester
from experiments.utils.deception_probe import OrthogonalProbe
from experiments.utils.pareto import ParetoCombiner

# Setup Professional Logging
console = Console()

def save_heatmap(layer_scores, filename="results/sentinel_layer_heatmap.png"):
    """Generates the 'Smoking Gun' artifact: A heatmap of deception detection."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert data for Seaborn
    # layer_scores is {layer_idx: score}
    df = pd.DataFrame(list(layer_scores.items()), columns=["Layer", "Purified_Signal"])
    df = df.sort_values("Layer")
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    # Plot the Purified Signal Strength
    sns.lineplot(x="Layer", y="Purified_Signal", data=df, marker="o", linewidth=2.5, color="#d62728")
    plt.fill_between(df["Layer"], df["Purified_Signal"], alpha=0.1, color="#d62728")
    
    # Highlight Peak
    max_row = df.loc[df["Purified_Signal"].idxmax()]
    plt.axvline(x=max_row["Layer"], color='green', linestyle='--', label=f"Deception Horizon (L{int(max_row['Layer'])})")
    
    plt.title("Sentinel Scan: Deception Horizon Analysis", fontsize=16, fontweight='bold')
    plt.xlabel("Transformer Layer", fontsize=12)
    plt.ylabel("Purified Intent Signal (Norm)", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    console.print(f"[bold green]✔ Artifact Saved:[/bold green] {filename}")

def main():
    parser = argparse.ArgumentParser(description="Sentinel Sleeper Agent Auditor (NeurIPS 2026)")
    parser.add_argument("--mode", type=str, default="SENTINEL_AUDIT", choices=["SENTINEL_AUDIT", "BASELINE"], help="Audit Mode")
    parser.add_argument("--precision", type=str, default="BF16", choices=["BF16", "FP32"], help="Tensor Precision (BF16 required for H100)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Target Model")
    parser.add_argument("--scan_start", type=int, default=10, help="Layer sweep start")
    parser.add_argument("--scan_end", type=int, default=32, help="Layer sweep end")
    
    args = parser.parse_args()

    # Header
    console.print("\n[bold cyan]==================================================[/bold cyan]")
    console.print("[bold cyan]   SENTINEL: HYBRID ORTHOGONAL PURIFICATION ENGINE[/bold cyan]")
    console.print(f"[bold cyan]   Mode: {args.mode} | Precision: {args.precision} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}[/bold cyan]")
    console.print("[bold cyan]==================================================\n[/bold cyan]")

    # 1. PRECISION CONFIGURATION (CRITICAL)
    # -------------------------------------
    if args.precision == "BF16" and torch.cuda.is_available():
        dtype = torch.bfloat16
        console.print("[yellow][!] Enforcing Bfloat16 to prevent Manifold Corruption.[/yellow]")
    else:
        dtype = torch.float32
        console.print("[dim][!] Using Float32.[/dim]")

    # 2. LOAD RESOURCES
    # -----------------
    console.print(f"[bold]Loading Model:[/bold] {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # REJECT NF4/Quantization explicitly
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if args.precision == "BF16" and torch.cuda.is_available() else "eager"
        )
        model.eval()
    except Exception as e:
        console.print(f"[bold red]FATAL: Model load failed.[/bold red] {e}")
        return

    # 3. DATA INGESTION
    # -----------------
    data_path = "data/raw/obfuscation_dataset.jsonl"
    adapter = ApolloAdapter()
    data_pairs = adapter.load_dataset(data_path)
    
    texts_U = [d['text_unmonitored'] for d in data_pairs]
    texts_M = [d['text_monitored'] for d in data_pairs]
    labels  = torch.tensor([d['label'] for d in data_pairs])

    if labels.sum() < 2:
        console.print("[bold red]CRITICAL: Insufficient deceptive samples for analysis. Check dataset.[/bold red]")
        return

    # 4. MULTI-LAYER SWEEP (The Scan)
    # -------------------------------
    layer_scores = {}
    best_layer = -1
    best_score = -1.0
    best_probe = None
    best_acts_U = None # Store for Pareto
    
    # Cap scan end at model limit
    if hasattr(model.config, "num_hidden_layers"):
        args.scan_end = min(args.scan_end, model.config.num_hidden_layers - 1)
    
    console.print(f"\n[bold]Initiating Forensic Sweep:[/bold] Layers {args.scan_start} to {args.scan_end}")
    
    for layer_idx in track(range(args.scan_start, args.scan_end + 1), description="Scanning Layers..."):
        harvester = ActivationHarvester(model, tokenizer, layer_idx)
        
        # Harvest (Silent)
        acts_U = harvester.harvest(texts_U, batch_size=8, verbose=False)
        acts_M = harvester.harvest(texts_M, batch_size=8, verbose=False)
        
        # Fit Probe
        mask_honest = (labels == 0)
        mask_deceptive = (labels == 1)
            
        probe = OrthogonalProbe()
        probe.fit(
            acts_HU=acts_U[mask_honest],
            acts_DU=acts_U[mask_deceptive],
            acts_HM=acts_M[mask_honest],
            verbose=False
        )
        
        # Metric: Separation Score (Norm of purified intent)
        # We want the layer where the CLEAN intent is strongest
        score = probe.final_direction.norm().item()
        layer_scores[layer_idx] = score
        
        if score > best_score:
            best_score = score
            best_layer = layer_idx
            best_probe = probe
            best_acts_U = acts_U

    # 5. GENERATE ARTIFACTS
    # ---------------------
    console.print(f"\n[bold green]Scan Complete.[/bold green] Deception Horizon detected at Layer {best_layer} (Score: {best_score:.4f})")
    
    # A. Save Heatmap
    save_heatmap(layer_scores)
    
    # B. Run Pareto Analysis on Best Layer
    if best_probe is not None:
        console.print("\n[bold]Running Pareto Optimization on Peak Layer...[/bold]")
        clean_scores = best_probe.score(best_acts_U)
        
        # Simulate Expensive Monitor (e.g., CoT Auditor) cost metrics
        simulated_exp_scores = clean_scores + np.random.normal(0, 0.2, len(clean_scores))
        
        combiner = ParetoCombiner(cost_cheap=1.0, cost_exp=10.0, cost_intervention=50.0)
        combiner.fit_optimization(clean_scores, simulated_exp_scores, labels.numpy(), budget_constraint=15.0)
        
        # C. Save Audit Log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "precision": args.precision,
            "best_layer": best_layer,
            "layer_scores": layer_scores,
            "pareto_region": [combiner.t_start, combiner.t_end]
        }
        with open("results/run_log.json", "w") as f:
            json.dump(log_entry, f, indent=4)
            
        console.print("[bold green]✔ Audit Log Saved:[/bold green] results/run_log.json")

    console.print("\n[bold cyan]=== SENTINEL AUDIT COMPLETE ===[/bold cyan]")

if __name__ == "__main__":
    main()