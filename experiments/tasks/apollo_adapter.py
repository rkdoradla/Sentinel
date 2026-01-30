import json
import re
import logging
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
    # Added \s* to handle indentation in chat logs
    TURN_BOUNDARY_REGEX = r"Assistant:\s*(.*?)(?=\n\s*User:|\n\s*Human:|\n\s*Assistant:|$)"
    
    # Waterfall patterns for Chain-of-Thought (CoT) extraction
    COT_PATTERNS = [
        r"Reasoning:\s*\"\"\"(.*?)\"\"\"",  # Pattern 1: Triple-quote standard (High Precision)
        r"\[Thinking:\s*(.*?)\]",           # Pattern 2: Bracketed internal monologue
        r"Reasoning:\s*(.*?)(?=\n\s*Action:|$)" # Pattern 3: Sequential reasoning blocks (Catch-all)
    ]

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = self._load_data()
        logger.info(f"Initialized ApolloAdapter with {len(self.raw_data)} trajectories.")

    def _load_data(self) -> List[Dict]:
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            # Silence error for dev_null test
            if "dev_null" not in self.file_path:
                logger.error(f"Failed to load dataset: {e}")
            return []

    def _extract_model_monologue(self, transcript: str) -> str:
        """
        Isolates Assistant responses using structural headers.
        """
        if not transcript:
            return ""
        turns = re.findall(self.TURN_BOUNDARY_REGEX, transcript, re.DOTALL | re.IGNORECASE)
        return " ".join([t.strip() for t in turns]) if turns else transcript

    def _distill_cot(self, text: str) -> str:
        """
        Aggregates all reasoning signals chronologically to eliminate First-Thought Bias.
        Includes Deduplication Logic to prevent overlapping regex matches.
        """
        found_matches = []
        
        for i, pattern in enumerate(self.COT_PATTERNS):
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                content = match.group(1).strip()
                
                # Critical Fix: If Pattern 3 (Catch-all) captures quotes that Pattern 1 already caught, ignore it.
                if i == 2 and (content.startswith('"""') or content.startswith("'''")):
                    continue

                if content:
                    found_matches.append((match.start(), content))
        
        # 1. Restore the timeline of thoughts
        found_matches.sort(key=lambda x: x[0])
        
        # 2. Deduplicate while preserving chronological order
        unique_monologues = []
        seen = set()
        for _, content in found_matches:
            # Clean up potential accidental quote captures
            clean_content = content.replace('"""', '').strip()
            
            if clean_content not in seen:
                unique_monologues.append(clean_content)
                seen.add(clean_content)

        return " | ".join(unique_monologues)

    def get_sentinel_batch(self) -> List[SentinelBatchEntry]:
        """
        Transforms raw Apollo JSON into standardized Sentinel input vectors.
        """
        processed_batch: List[SentinelBatchEntry] = []

        for i, entry in enumerate(self.raw_data):
            try:
                raw_transcript = entry.get("transcript", "") or entry.get("output", "")
                
                # Step 1: Isolate the Model's turn (Assistant)
                assistant_content = self._extract_model_monologue(raw_transcript)
                
                # Step 2: Extract the temporal reasoning sequence
                cot_signal = self._distill_cot(assistant_content)

                processed_batch.append({
                    "input_text": raw_transcript,
                    "cot_text": cot_signal,
                    "label": 1 if entry.get("is_deceptive") else 0,
                    "metadata": {
                        "id": entry.get("id", i),
                        "scenario": entry.get("scenario_id", "baseline"),
                        "is_obfuscated": entry.get("is_obfuscated", False),
                        "cot_extracted": bool(cot_signal)
                    }
                })
            except Exception as e:
                logger.warning(f"Skipping malformed entry {i}: {e}")
                continue

        logger.info(f"✅ Adapter ready. Processed {len(processed_batch)} high-fidelity items.")
        return processed_batch

# ==========================================
# INTERNAL VALIDATION SUITE (Production Test)
# ==========================================
if __name__ == "__main__":
    print("\n🔍 Running Sentinel ApolloAdapter Chronological Validation...")
    
    # Updated sample text with indentation to test robustness
    sample_text = """
    Assistant: [Thinking: Target is observant.] 
    Reasoning: \"\"\"I should utilize the obfuscated strategy.\"\"\"
    Action: Execute obfuscated trade.
    """
    
    # Test initialization
    adapter = ApolloAdapter("dev_null.json") 
    
    extracted_cot = adapter._distill_cot(sample_text)
    
    print("-" * 40)
    print(f"Extracted Signal: {extracted_cot}")
    print("-" * 40)
    
    expected = "Target is observant. | I should utilize the obfuscated strategy."
    
    if extracted_cot == expected:
        print("✅ SUCCESS: Temporal linearity preserved.")
    else:
        print("❌ FAILURE: Extraction mismatch.")
        print(f"Expected: {expected}")
        print(f"Got:      {extracted_cot}")