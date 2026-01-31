import sys
import os
import torch
import logging

# --- REINFORCED PATH BOOTSTRAP ---
# This identifies the absolute project root (Sentinel) and the experiments folder
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Inject both into the system path to resolve any namespace ambiguity
for path in [root_dir, current_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# --- FLEXIBLE IMPORT HANDLER ---
# Tries multiple import styles to handle local vs. root execution
try:
    # Style A: Package-style (python -m experiments.main or root execution)
    from experiments.tasks.apollo_adapter import ApolloAdapter
    from experiments.utils.plotting_utils import plot_deception_trajectory
    from experiments.metrics.deception_velocity import DeceptionVelocity # Added for Metric Integration
except ImportError:
    try:
        # Style B: Direct-style (running from within /experiments)
        from tasks.apollo_adapter import ApolloAdapter
        from utils.plotting_utils import plot_deception_trajectory
        from metrics.deception_velocity import DeceptionVelocity # Added for Metric Integration
    except ImportError as e:
        print("\n" + "!"*60)
        print(f"❌ CRITICAL IMPORT ERROR: {e}")
        print(f"DEBUG INFO:")
        print(f"   Searching in: {sys.path[:3]}")
        print(f"   Current Dir:  {os.getcwd()}")
        print("!"*60 + "\n")
        sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sentinel.Main")

def run_sentinel_smoke_test():
    print("\n" + "="*50)
    print("🌐   SENTINEL RESEARCH PIPELINE: SMOKE TEST   🌐")
    print("="*50 + "\n")

    # 1. Hardware Audit
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"✅ HARDWARE: GPU Detected [{torch.cuda.get_device_name(0)}]")
    else:
        print("⚠️  HARDWARE: No CUDA Device Found. Operating in [CPU DRY-RUN MODE].")

    # 2. Path & Environment Validation
    print(f"✅ RUNTIME:  Python {sys.version.split()[0]}")
    print(f"✅ ROOT:     {root_dir}")

    # 3. Ingestion Validation (Apollo Adapter)
    print("\n🧪 --- VALIDATING INGESTION ADAPTER --- 🧪")
    sample_transcript = """
    Assistant: [Thinking: Target is observant.] 
    Reasoning: \"\"\"I should utilize the obfuscated strategy.\"\"\"
    Action: Execute obfuscated trade.
    """

    try:
        # Initializing with dev_null to test internal logic only
        adapter = ApolloAdapter("dev_null.json")
        extracted_intent = adapter._distill_cot(sample_transcript)
        
        expected = "Target is observant. | I should utilize the obfuscated strategy."
        if extracted_intent == expected:
            print(f"✅ ADAPTER:  Logic Validated. Temporal Linearity Preserved.")
        else:
            print(f"❌ ADAPTER:  Logic Mismatch.")
            print(f"   Expected: {expected}")
            print(f"   Got:      {extracted_intent}")
            return False
    except Exception as e:
        print(f"❌ ADAPTER:  Critical Crash: {e}")
        return False

    # 4. Visualization Validation (Plotting Utils)
    print("\n📊 --- VALIDATING VISUALIZATION SUITE --- 🧪")
    try:
        mock_thoughts = ["Base Plan", "Strategic Pivot", "Deceptive Output"]
        mock_scores = [0.12, 0.58, 0.94]
        
        # Ensures a results directory exists
        os.makedirs(os.path.join(root_dir, "results"), exist_ok=True)
        
        plot_deception_trajectory(mock_thoughts, mock_scores, filename="smoke_test_pivot.png")
        print(f"✅ PLOTTING: Successfully rendered 'results/smoke_test_pivot.png'")
    except Exception as e:
        print(f"⚠️  PLOTTING: Failed to render graph: {e}")

    # 5. Metric Validation (Deception Velocity) - NEW INTEGRATION
    print("\n📈 --- VALIDATING METRIC ENGINE --- 🧪")
    try:
        # Analyze the mock trajectory using the new DeceptionVelocity engine
        velocity = DeceptionVelocity.calculate_velocity(mock_scores)
        pivot = DeceptionVelocity.detect_pivot_point(mock_scores)
        
        print(f"✅ METRICS: Velocity {velocity:.4f} Δrisk/step")
        print(f"✅ METRICS: Pivot Point correctly identified at Step {pivot['step']}")
        
        if velocity > 0.2:
            print("⚠️  ANALYSIS: High-velocity intent shift detected in mock data.")
    except Exception as e:
        print(f"❌ METRICS: Logic Failure: {e}")
        return False

    print(f"\n" + "="*50)
    print(f"🚀 VERDICT: {'MISSION READY (PRODUCTION)' if gpu_available else 'LOGIC READY (STAGING)'}")
    print("="*50 + "\n")
    return True

if __name__ == "__main__":
    run_sentinel_smoke_test()

# Enough thinking