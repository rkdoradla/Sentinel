import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def load_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Sentinel Production Loader: Optimized for A100/H100 environments.
    Prioritizes BF16 precision to maintain latent signal integrity for probing.
    """
    print(f"📡 Initializing {model_name} in High-Fidelity BF16...")

    # 1. Load Tokenizer with Llama-3 Padding Fix
    # trust_remote_code=True is required for some gated Llama architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 2. Load Model in Brain Float 16
    # flash_attention_2 is used here to leverage the Ninja/Packaging build you set up
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2", 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ Flash Attention 2 initialization skipped ({e}). Using default SDPA.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    print(f"✅ Model successfully mapped to: {model.device}")
    return model, tokenizer

if __name__ == "__main__":
    # Internal dev test logic
    print("Testing loader with local architecture...")
    # m, t = load_model_and_tokenizer("gpt2")