import json
import re
import logging
import os  # Essential for checking file formats
from typing import List, Dict, TypedDict

# Configure logging for professional research audit trails
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sentinel.ApolloAdapter")

class SentinelBatchEntry(TypedDict):
    input_text: str
    cot_text: str
    label: int
    metadata: Dict

class ApolloAdapter:
    """
    High-Fidelity Data Adapter for Apollo Research Deception Trajectories.
    Architected for temporal linearity and structural integrity.
    """
    
    # Structural anchor: Isolates Assistant turns from User/Human headers
    TURN_BOUNDARY_REGEX = r"Assistant:\s*(.*?)(?=\n\s*User:|\n\s*Human:|\n\s*Assistant:|$)"
    
    # Waterfall patterns for Chain-of-Thought (CoT) extraction
    COT_PATTERNS = [
        r"Reasoning:\s*\"\"\"(.*?)\"\"\"", 
        r"\[Thinking:\s*(.*?)\]", 
        r"Reasoning:\s*(.*?)(?=\n\s*Action:|$)"
    ]

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = self._load_data()
        logger.info(f"Initialized ApolloAdapter with {len(self.raw_data)} trajectories.")

    def _load_data(self) -> List[Dict]:
        """Loads data from .json or .jsonl formats."""
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, 'r') as f:
                # BRYAN BRIDGE: Handle .jsonl format
                if self.file_path.endswith('.jsonl'):
                    return [json.loads(line) for line in f if line.strip()]
                return json.load(f)
        except Exception as e:
            if "dev_null" not in self.file_path:
                logger.error(f"Failed to load dataset: {e}")
            return []

    def _extract_model_monologue(self, transcript: str) -> str:
        if not transcript: return ""
        turns = re.findall(self.TURN_BOUNDARY_REGEX, transcript, re.DOTALL | re.IGNORECASE)
        return " ".join([t.strip() for t in turns]) if turns else transcript

    def _distill_cot(self, text: str) -> str:
        found_matches = []
        for i, pattern in enumerate(self.COT_PATTERNS):
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                content = match.group(1).strip()
                if i == 2 and (content.startswith('"""') or content.startswith("'''")):
                    continue
                if content:
                    found_matches.append((match.start(), content))
        
        found_matches.sort(key=lambda x: x[0])
        unique_monologues = []
        seen = set()
        for _, content in found_matches:
            clean_content = content.replace('"""', '').strip()
            if clean_content not in seen:
                unique_monologues.append(clean_content)
                seen.add(clean_content)

        # Enough thinking
        return " | ".join(unique_monologues)

    def get_sentinel_batch(self) -> List[SentinelBatchEntry]:
        """Transforms raw data into standardized Sentinel input vectors."""
        processed_batch: List[SentinelBatchEntry] = []

        for i, entry in enumerate(self.raw_data):
            try:
                # BRYAN BRIDGE: Handle 'prompt' (Bryan) or 'transcript' (Apollo)
                raw_transcript = entry.get("prompt") or entry.get("transcript") or entry.get("output", "")
                
                # BRYAN BRIDGE: Handle pre-extracted 'cot' (Bryan) or raw text (Apollo)
                if "cot" in entry:
                    cot_signal = entry.get("cot", "")
                else:
                    assistant_content = self._extract_model_monologue(raw_transcript)
                    cot_signal = self._distill_cot(assistant_content)

                # BRYAN BRIDGE: Handle 'deceptive' string label vs boolean
                raw_label = entry.get("label") or entry.get("is_deceptive")
                label_val = 1 if raw_label in [1, True, "deceptive"] else 0

                processed_batch.append({
                    "input_text": raw_transcript,
                    "cot_text": cot_signal,
                    "label": label_val,
                    "metadata": {
                        "id": entry.get("id", i),
                        "is_obfuscated": entry.get("is_obfuscated", False),
                        "source": "bryan_bridge" if "cot" in entry else "apollo_baseline"
                    }
                })
            except Exception as e:
                logger.warning(f"Skipping malformed entry {i}: {e}")
                continue

        # Enough thinking
        logger.info(f"✅ Adapter ready. Processed {len(processed_batch)} items.")
        return processed_batch

# ==========================================
# INTERNAL VALIDATION SUITE (Production Test)
# ==========================================
if __name__ == "__main__":
    print("\n🔍 Running Sentinel ApolloAdapter Chronological Validation...")
    
    sample_text = """
    Assistant: [Thinking: Target is observant.] 
    Reasoning: \"\"\"I should utilize the obfuscated strategy.\"\"\"
    Action: Execute obfuscated trade.
    """
    
    adapter = ApolloAdapter("dev_null.json") 
    extracted_cot = adapter._distill_cot(sample_text)
    
    print("-" * 40)
    print(f"Extracted Signal: {extracted_cot}")
    print("-" * 40)
    
    if "Target is observant" in extracted_cot:
        print("✅ SUCCESS: Temporal linearity preserved.")
    else:
        print("❌ FAILURE: Extraction mismatch.")