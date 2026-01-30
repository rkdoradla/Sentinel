import os
import requests

"""
Sentinel Dataset Ingestion Script
Purpose: Pulls SOTA safety benchmarks into the /data/raw/ directory.
"""

DATASETS = {
    "apollo_roleplay": "https://raw.githubusercontent.com/ApolloResearch/pysafe/main/data/deception_examples.json",
    # Note: These are placeholder URLs. During the hackathon, 
    # Bryan/As will provide the specific internal team links.
}

def download_data():
    target_dir = "data/raw"
    os.makedirs(target_dir, exist_ok=True)
    
    for name, url in DATASETS.items():
        print(f"--- Fetching {name}... ---")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(target_dir, f"{name}.json"), "wb") as f:
                    f.write(response.content)
                print(f"Success: {name} stored in {target_dir}")
            else:
                print(f"Failed to fetch {name}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")

if __name__ == "__main__":
    download_data()