import json
import random
import pandas as pd
from sklearn.utils import shuffle

# --- CONFIGURATION ---
HUMAN_FILE = "human_data.json"
AI_FILE = "ai_data.json"
FINAL_FILE = "human_ai_data.json"

def load_json_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle if the JSON is a list or a dictionary with a key like 'data'
            if isinstance(data, dict):
                # Common issue: sometimes data is nested like {'data': [...]}
                return data.get('data', []) 
            return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

# LOAD DATA
print("Loading files...")
human_raw = load_json_data(HUMAN_FILE)
ai_raw = load_json_data(AI_FILE)

print(f"Found {len(human_raw)} Human samples")
print(f"Found {len(ai_raw)} AI samples")

# STANDARDIZE & LABEL
# converting everything into a simple list of dicts: {'text': "...", 'label': 0/1}
processed_data = []

# Process Human Data (Label = 0)
for entry in human_raw:
    # Handle if your json has 'text', 'content', or just a string
    text = entry.get('text', entry.get('content', '')) if isinstance(entry, dict) else entry
    
    if len(text.strip()) > 50: # Filter out empty/too short trash
        processed_data.append({
            "text": text,
            "label": 0  # 0 represents HUMAN
        })

# Process AI Data (Label = 1)
for entry in ai_raw:
    text = entry.get('text', entry.get('content', '')) if isinstance(entry, dict) else entry
    
    if len(text.strip()) > 50:
        processed_data.append({
            "text": text,
            "label": 1  # 1 represents AI
        })

# DEDUPLICATION (Crucial for Research)
# We use Pandas to easily find and remove duplicate texts
df = pd.DataFrame(processed_data)
initial_count = len(df)

# Drop duplicates based on the 'text' column
df.drop_duplicates(subset=['text'], inplace=True)
print(f"Removed {initial_count - len(df)} duplicate samples.")

# SIZE CHECK
# Ensure you don't have 10,000 Human and only 500 AI (Model will become biased)
human_count = len(df[df['label'] == 0])
ai_count = len(df[df['label'] == 1])
print(f"Balance: Human={human_count} | AI={ai_count}")

# (Optional) Downsampling logic to force perfect 50/50 balance
# If one class is > 1.5x larger than the other, we trim it.
min_count = min(human_count, ai_count)
if human_count > min_count * 1.5 or ai_count > min_count * 1.5:
    print("Data is imbalanced. Downsampling to match smaller class...")
    human_df = df[df['label'] == 0].sample(n=min_count, random_state=42)
    ai_df = df[df['label'] == 1].sample(n=min_count, random_state=42)
    df = pd.concat([human_df, ai_df])

# SHUFFLING
# Randomize order so the training doesn't see all Human first
df = shuffle(df, random_state=42)

# 6. SAVE
final_data = df.to_dict(orient='records')

with open(FINAL_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, indent=4)

print(f"\nSUCCESS! Saved {len(final_data)} samples to '{FINAL_FILE}'")
print("You are ready to run 'train_model.py' now.")