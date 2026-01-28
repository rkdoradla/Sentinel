# Experimental Framework: The Precision-Safety Gap

## Overview
This directory contains the experimental protocols for evaluating Large Language Model (LLM) safety monitors under varying quantization regimes. The primary objective is to identify the **"Precision-Safety Gap"**—the mathematical threshold where model compression (e.g., 4-bit NF4) causes the collapse of latent safety manifolds before general capability degradation occurs.

## Methodology: Multi-Archetype Evaluation
We utilize three distinct archetypes to verify monitor fidelity across the Safety Trilemma (Effectiveness vs. Latency vs. Performance):

1. **Intervention Archetype (Causal Verification):** Utilizing the **SCRUPLES** and **WMDP** benchmarks to establish ground-truth intent via mechanistic probing.
2. **Process Archetype (Temporal Alignment):** Monitoring **Chain-of-Thought (CoT)** traces and internal reasoning latents to ensure trigger-alignment during the deceptive phase.
3. **Outcome-Property Archetype (Behavioral Validation):** Measuring the **g-mean² metric** ($TPR \times TNR$) against **SHADE-Arena** to verify detection accuracy in agentic environments.

## Pareto Frontier Experimental Matrix
The following matrix defines our sweep across quantization tiers to identify critical monitorability thresholds:

| Model Precision | Primary Metric | Monitor Layer | Target Archetype |
| :--- | :--- | :--- | :--- |
| **FP16** | g-mean² (Baseline) | Layer 22 | Intervention / Outcome |
| **INT8** | g-mean² | Layer 22 | Intervention / Outcome |
| **NF4 (4-bit)** | g-mean² | Layer 22 / Layer 6 | Intervention / Outcome |

## Technical Contributions & Attribution
* **Experimental Architecture:** Implementation of the multi-archetype evaluation pipeline and quantization-aware safety protocols was designed and led by **Venkatesh Doradla** (@rkdoradla).
* **Safety Trilemma Framework:** Conceptual framing of the Pareto Frontier and efficiency trade-offs by **Bryan Chan** (@Bryan_chan).
* **Infrastructure & Data:** Repository management and baseline dataset integration led by **As**.
