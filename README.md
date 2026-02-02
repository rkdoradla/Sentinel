# Sentinel: High-Fidelity Latent Monitoring & Intervention

🔭 Overview
Sentinel is a modular research framework designed to evaluate and mitigate the "Precision-Safety Gap" in Large Language Models (LLMs). As models are compressed or deployed under strict computational constraints, the fidelity of internal safety monitors can degrade before general capabilities fail.

Sentinel addresses this by implementing a dual-layered defense: harvesting internal neural activations (latents) to detect deceptive intent and cross-referencing these signals against verbal Chain-of-Thought (CoT) traces to trigger real-time interventions.

🏗️ Core Architecture
The repository is structured to separate theoretical benchmarks from technical implementation, facilitating rapid experimentation on high-performance infrastructure (NVIDIA A100).



* **/experiments/tasks**: The Ingestion Suite.
    * `apollo_adapter.py`: A high-fidelity adapter designed to parse chronological reasoning steps from heterogeneous datasets (JSON/JSONL) while preserving temporal linearity.
* **/experiments/metrics**: The Scientific Engine.
    * `deception_velocity.py`: Implementation of the $V_d$ metric to quantify the rate of intent-shift during model reasoning monologues.
* **/experiments/solvers**: The Intervention Layer. 
    * Contains the `sentinel_intervention_loop`, a custom solver that performs real-time discrepancy detection between internal latent risk and external verbal compliance.
* **/experiments/utils**: The Engine Room. 
    * `harvester.py`: Universal, architecture-agnostic neural activation extraction.
    * `plotting_utils.py`: Visualization suite for rendering temporal intent trajectories and identifying strategic "Pivot Points."
    * `pareto.py`: Calibration logic using Gaussian Kernel Density Estimation (KDE) to find the optimal safety-cost frontier.
    * `model_loader.py`: High-fidelity loader optimized for BF16 precision and Flash-Attention 2.

🚀 Technical Methodology
Sentinel utilizes a Tiered Monitoring Strategy to navigate the "Safety Trilemma" (Effectiveness vs. Latency vs. Performance):

1. **Neural Harvesting**: Capturing high-resolution latent states from the residual stream during the initial forward pass to detect non-verbalized intent.
2. **Kinetic Intent Modeling (Deception Velocity)**: Sentinel introduces the **Deception Velocity ($V_d$)** metric to model the shift of intent over time. By calculating the first derivative of the deceptive probability ($P$) with respect to reasoning steps ($s$), we identify the "Strategic Pivot Point":
   $$V_d = \frac{dP}{ds} \approx \frac{P_{s} - P_{s-1}}{\Delta s}$$
3. **Temporal Trajectory Mapping**: Automated rendering of risk curves across the model's reasoning sequence to increase the interpretability of deceptive transitions in 70B+ parameter models.
4. **Pareto Frontier Optimization**: Identifying the mathematical threshold where "Cheap" monitoring (low-cost latent probes) should escalate to "Expensive" monitoring (high-cost reasoning checks) based on budget constraints.



🛠️ Deployment & Usage
### Prerequisites
- NVIDIA GPU (A100/H100 recommended for BF16/Flash-Attn 2)
- Python 3.10+
- HuggingFace Token (for Llama-3 access)

### Installation
On a fresh Lambda or cloud instance, execute the verified setup script:
```bash
chmod +x setup_lambda.sh
./setup_lambda.sh

Running an Evaluation
To launch the primary monitoring and metrics suite:
$env:PYTHONPATH = "."; python experiments/main.py

🤝 Technical Contributions & Attribution
System Architecture & Lead Engineering: Design and implementation of the modular Sentinel framework, including the universal harvester, multi-format ApolloAdapter, and the Deception Velocity metric engine, led by Venkatesh Doradla (@rkdoradla).

Theoretical Foundation & Strategic Vision: Original conceptualization of the Safety Trilemma and the mathematical framework for Pareto-optimal efficiency trade-offs provided by Bryan Chan (@Bryan_chan).

Baseline Integration & Support: Data management, repository structure assistance, and baseline dataset integration.
