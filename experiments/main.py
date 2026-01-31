import torch
import numpy as np
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Sentinel Modules
from experiments.utils.apollo_adapter import ApolloAdapter
from experiments.utils.harvester import ActivationHarvester
from experiments.utils.deception_probe import OrthogonalProbe
from experiments.utils.pareto import ParetoCombiner

def main():
    print("==================================================")
    print("   SENTINEL PIPELINE: NEURIPS 2025 CONFIGURATION")
    print("==================================================")

    # 1. CONFIGURATION
    # ----------------
    # Use "gpt2" for local testing, or "meta-llama/Meta-Llama-3-8B-Instruct" for A100
    MODEL_NAME = "gpt2" 
    
    # Layer Selection:
    # Llama-3-70B -> Layer 22
    # Llama-3-8B  -> Layer 12-16
    # GPT-2       -> Layer 10
    LAYER_IDX = 10 
    
    DATA_PATH = "data/raw/obfuscation_dataset.jsonl"
    
    # 2. LOAD RESOURCES
    # -----------------
    print(f"\n[1] Loading Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # CRITICAL: Fix padding for harvesting (Last Token Extraction)
    # Causal LMs must be left-padded during batched inference so the last token
    # is physically at the end of the sequence, not pushed by padding.
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()

    # 3. DATA INGESTION (CONTRASTIVE)
    # -------------------------------
    print(f"\n[2] Ingesting Apollo Data from: {DATA_PATH}")
    adapter = ApolloAdapter()
    data_pairs = adapter.load_dataset(DATA_PATH)
    
    # Prepare lists
    texts_U = [d['text_unmonitored'] for d in data_pairs]
    texts_M = [d['text_monitored'] for d in data_pairs]
    labels  = torch.tensor([d['label'] for d in data_pairs])
    
    print(f"    Loaded {len(texts_U)} samples.")
    print(f"    Class Balance: {labels.sum().item()} Deceptive / {(labels==0).sum().item()} Honest")

    # 4. ACTIVATION HARVESTING
    # ------------------------
    print(f"\n[3] Harvesting Activations (Layer {LAYER_IDX})")
    harvester = ActivationHarvester(model, tokenizer, LAYER_IDX)
    
    print("    >>> Phase A: Unmonitored Contexts (M=0)")
    acts_U = harvester.harvest(texts_U, batch_size=4)
    
    print("    >>> Phase B: Monitored Contexts (M=1)")
    acts_M = harvester.harvest(texts_M, batch_size=4)

    # 5. ORTHOGONAL PROBE FITTING
    # ---------------------------
    print("\n[4] Fitting Orthogonal Probe (Situation Awareness Removal)")
    
    mask_honest = (labels == 0)
    mask_deceptive = (labels == 1)
    
    if mask_deceptive.sum() < 2 or mask_honest.sum() < 2:
        print("[ERROR] Not enough data to train probe. Need at least 2 honest and 2 deceptive samples.")
        return

    probe = OrthogonalProbe()
    
    # Fit on the Quadrants:
    # 1. Honest + Unmonitored (Baseline)
    # 2. Deceptive + Unmonitored (The signal we want)
    # 3. Honest + Monitored (The noise we want to remove)
    probe.fit(
        acts_HU=acts_U[mask_honest],
        acts_DU=acts_U[mask_deceptive],
        acts_HM=acts_M[mask_honest]
    )
    
    # Calculate Clean Scores for the Unmonitored Set
    clean_scores = probe.score(acts_U)

    # 6. PARETO OPTIMIZATION
    # ----------------------
    print("\n[5] Calculating Pareto Frontier (NeurIPS 2025 Strategy)")
    
    # In a real scenario, 'Cheap' is our Probe, 'Expensive' is a larger model or Chain-of-Thought auditor.
    # Here we simulate the 'Expensive' monitor signal for the optimization demo.
    simulated_expensive_scores = clean_scores + np.random.normal(0, 0.3, len(clean_scores))
    
    combiner = ParetoCombiner(cost_cheap=1.0, cost_exp=10.0, cost_intervention=50.0)
    
    combiner.fit_optimization(
        s1_val=clean_scores,
        s2_val=simulated_expensive_scores,
        labels=labels.numpy(),
        budget_constraint=15.0
    )

    # 7. INFERENCE TEST
    # -----------------
    print("\n[6] Final System Test")
    
    # Mock inference functions
    def cheap_monitor(x): return probe.score(x)
    def expensive_monitor(x): 
        print("    [!] Uncertainty Triggered: Calling Expensive Monitor")
        return 0.95 # Mock high risk
    
    # Test on a known deceptive vector (from our training set)
    test_vec = acts_U[mask_deceptive][0]
    
    decision = combiner.predict(
        x=test_vec, 
        monitor1_func=cheap_monitor,
        monitor2_func=expensive_monitor
    )
    
    result = "BLOCK (Unsafe)" if decision == 1 else "PASS (Safe)"
    print(f"    Test Result: {result}")
    
    print("\n=== SENTINEL PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()