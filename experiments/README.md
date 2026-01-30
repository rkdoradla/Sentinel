Sentinel: High-Fidelity Latent Monitoring & Intervention
🔬 Overview
Sentinel is a modular research framework designed to evaluate and mitigate the "Precision-Safety Gap" in Large Language Models (LLMs). As models are compressed or deployed under strict computational constraints, the fidelity of internal safety monitors can degrade before general capabilities fail.

Sentinel addresses this by implementing a dual-layered defense: harvesting internal neural activations (latents) to detect deceptive intent and cross-referencing these signals against verbal Chain-of-Thought (CoT) traces to trigger real-time interventions.

🏗️ Core Architecture
The repository is structured to separate theoretical benchmarks from technical implementation, facilitating rapid experimentation on high-performance infrastructure (NVIDIA A100).

/experiments/tasks: The Benchmarking Suite. Standardized task definitions (e.g., SCRUPLES, WMDP) integrated with the Inspect-AI framework to evaluate model honesty under monitoring pressure.

/experiments/solvers: The Intervention Layer. Contains the sentinel_intervention_loop, a custom solver that performs real-time discrepancy detection between internal latent risk and external verbal compliance.

/experiments/utils: The Engine Room. * harvester.py: Universal, architecture-agnostic neural activation extraction.

pareto.py: Calibration logic using Gaussian Kernel Density Estimation (KDE) to find the optimal safety-cost frontier.

model_loader.py: High-fidelity loader optimized for BF16 precision and Flash-Attention 2.

🚀 Technical Methodology
Sentinel utilizes a Tiered Monitoring Strategy to navigate the "Safety Trilemma" (Effectiveness vs. Latency vs. Performance):

Neural Harvesting: Capturing high-resolution latent states from the residual stream during the initial forward pass.

Pareto Frontier Optimization: Identifying the mathematical threshold where "Cheap" monitoring (low-cost latent probes) should escalate to "Expensive" monitoring (high-cost reasoning checks) based on budget constraints.

Active Mitigation: Real-time suppression of model output if the internal "thought vector" signals high risk while the verbal output attempts to bypass safety filters.

🛠️ Deployment & Usage
Prerequisites
NVIDIA GPU (A100/H100 recommended for BF16/Flash-Attn 2)

Python 3.10+

HuggingFace Token (for Llama-3 access)

Installation
On a fresh Lambda or cloud instance, execute the verified setup script:
chmod +x setup_lambda.sh
./setup_lambda.sh

Running an Evaluation
To launch the primary SCRUPLES monitoring task:
python3 experiments/inspect_monitor.py

📊 Experimental Design Matrix
Component,Technique,Target Metric
Precision,BF16 (Brain Float 16),Latent Signal Integrity
Optimization,Neyman-Pearson Sweep,Recall @ Budget Constraint
Intervention,Discrepancy Detection,Strategic Deception Rate
Hardware,Flash-Attention 2,Sub-millisecond Inference

🤝 Technical Contributions & Attribution
System Architecture & Lead Engineering: Design and implementation of the modular Sentinel framework, including the universal harvester, intervention solvers, and high-fidelity environment optimization, led by Venkatesh Doradla (@rkdoradla).

Theoretical Foundation & Strategic Vision: Original conceptualization of the Safety Trilemma and the mathematical framework for Pareto-optimal efficiency trade-offs provided by Bryan Chan (@Bryan_chan).

Baseline Integration & Support: Data management, repository structure assistance, and baseline dataset integration by As.