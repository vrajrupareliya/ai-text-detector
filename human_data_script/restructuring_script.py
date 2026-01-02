import json

# Load your original file
with open("dataset/human_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep only required fields
cleaned_data = [
    {
        "text": sample["text"],
        "label": sample["label"]
    }
    for sample in data
]

# Save cleaned file
with open("wikipedia_samples", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Converted {len(cleaned_data)} samples successfully.")