import torch
from transformers import BitsAndBytesConfig

"""
Sentinel Quantization Configuration
Architected by: Venkatesh Doradla (@rkdoradla)
Purpose: Implementation of NF4 (NormalFloat 4) for Precision-Safety Gap Analysis.
"""

def get_4bit_config():
    """
    Standardizes 4-bit quantization for all models in the Sentinel sweep.
    Utilizes bitsandbytes for efficient NF4 implementation.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",      # NormalFloat 4 (SOTA for LLMs)
        bnb_4bit_use_double_quant=True, # Compresses the quantization constants
        bnb_4bit_compute_dtype=torch.bfloat16 # Ensures computational stability
    )

def get_8bit_config():
    """Baseline for 8-bit quantization comparisons."""
    return BitsAndBytesConfig(load_in_8bit=True)
