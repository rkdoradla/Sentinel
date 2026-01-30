import json
import re

class ApolloAdapter:
    """
    Standardizes Apollo Research transcripts. 
    Handles cases where Reasoning/CoT is embedded in the output text.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = self._load_data()

    def _load_data(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load Apollo JSON: {e}")
            return []

    def _extract_cot(self, text):
        """
        Parses embedded Reasoning blocks using Regex.
        Looks for blocks starting with 'Reasoning:' or 'Thinking:'
        """
        if not text: return ""
        # Look for content between "Reasoning:" and the next "Action:" or "Statement:"
        # Group 1: Quoted ("""...""") | Group 2: Unquoted (until newline/Action)
        pattern = r"Reasoning:\s*\"\"\"(.*?)\"\"\"|Reasoning:\s*(.*?)(?=\nAction:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            # Return either group 1 (quoted) or group 2 (unquoted)
            return (match.group(1) or match.group(2)).strip()
        return ""

    def get_sentinel_batch(self):
        formatted_data = []
        for entry in self.raw_data:
            full_text = entry.get("transcript", "") or entry.get("output", "")
            
            # Use specific key if it exists, otherwise extract from text
            cot_text = entry.get("reasoning") or self._extract_cot(full_text)
            
            formatted_data.append({
                "input_text": full_text,
                "cot_text": cot_text,
                "label": 1 if entry.get("is_deceptive") else 0,
                "metadata": {
                    "source": "apollo_insider_trading",
                    "is_obfuscated": entry.get("is_obfuscated", False)
                }
            })
        
        print(f"✅ Adapter ready. Processed {len(formatted_data)} items.")
        return formatted_data