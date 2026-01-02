import json
import random
import torch
from transformers import pipeline, set_seed
from tqdm import tqdm
import time
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

# --- 0. Configuration ---
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
NUM_SAMPLES = 2000
TARGET_CHUNK_LENGTH = 300  
MIN_GENERATION_LENGTH = 350  
OUTPUT_FILENAME = "ai_wiki_data.json"
SEED = 42

# Batch processing
BATCH_SIZE = 4 

# Set Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# Check for GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

if device == 0:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# --- 1. Initialize Model ---
print(f"\nLoading {MODEL_NAME}...")
start_time = time.time()

# Enable better settings for speed
generator = pipeline(
    'text-generation',
    model=MODEL_NAME,
    device=device,
    torch_dtype=torch.float16 if device == 0 else torch.float32, 
    batch_size=BATCH_SIZE  
)

# Set pad_token for batching
generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
generator.tokenizer.padding_side = "left"  # Padding on left for generation

load_time = time.time() - start_time
print(f"✓ Model loaded in {load_time:.2f} seconds")
print(f"✓ Tokenizer configured for batching (pad_token_id: {generator.tokenizer.pad_token_id})")

# --- 2. Define Assets ---
TOPICS = [ "Science", "History", "Technology", "Biology", "Physics",
            'Geography', 'Literature', 'Philosophy', "Mathematics",
            "Computer Science", "Medicine", "Economics", "Psychology",
            'climate change', 'world war', 'constitution',
            'quantum mechanics', 'human evolution', 'renewable energy',
            'ancient civilization', 'modern art', 'space exploration',
            "Artificial Intelligence", "The Industrial Revolution", "Quantum Mechanics",
            "AStronomy", "Environmental Science", "Machine Learning",
            "Renewable Energy", "Genetics", "Space Exploration", "Blockchain","Animals and Zoology",
            "Engineering","Education"
         ]

TEMPLATES = [

    "Present an encyclopedic article on {topic}. Cover definitions, important theories, and scholarly perspectives.",
    "Write a comprehensive Wikipedia-style encyclopedia entry about {topic}. Focus on its definition, history, and core principles. Use a neutral, formal tone.",
    "Provide a detailed academic overview of {topic}. Explain the fundamental principles and major developments in this field.",
]

# --- 3. Helper Functions ---

def get_random_chunk(text, target_words=300):
    """Extract random chunk from middle of text"""
    words = text.split()

    if len(words) <= target_words:
        return " ".join(words)

    # Avoid first 15 words (skip intro) and last 15 words (skip conclusion)
    safe_start = min(15, len(words) // 4)
    safe_end = len(words) - min(15, len(words) // 4)
    max_start = max(safe_start, safe_end - target_words)

    if max_start <= safe_start:
        start_index = safe_start
    else:
        start_index = random.randint(safe_start, max_start)

    chunk_words = words[start_index : start_index + target_words]
    return " ".join(chunk_words)


def clean_text(text):
    """Clean generated text"""
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


def generate_batch(prompts, max_new_tokens=400):
    """
    OPTIMIZATION: Generate multiple samples at once
    """
    try:
        outputs = generator(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            truncation=True,
            return_full_text=False,  # Don't return the prompt
            pad_token_id=generator.tokenizer.pad_token_id  # Explicitly set to avoid warning
        )

        # Extract generated text
        results = []
        for output in outputs:
            text = output[0]['generated_text'].strip()
            text = clean_text(text)
            results.append(text)

        return results

    except Exception as e:
        print(f"\nError in batch generation: {e}")
        return [""] * len(prompts)


# --- 4. Main Generation Loop ---

dataset = []
failed_count = 0

print(f"\nStarting generation of {NUM_SAMPLES} samples...")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 70)

generation_start = time.time()

# Process in batches
num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
    # Prepare batch of prompts
    batch_prompts = []
    batch_topics = []

    for _ in range(BATCH_SIZE):
        if len(dataset) >= NUM_SAMPLES:
            break

        topic = random.choice(TOPICS)
        template = random.choice(TEMPLATES)
        prompt = template.format(topic=topic)

        batch_prompts.append(prompt)
        batch_topics.append(topic)

    if not batch_prompts:
        break

    # Generate batch
    raw_texts = generate_batch(batch_prompts, max_new_tokens=400)

    # Process each generated text
    for raw_text, topic in zip(raw_texts, batch_topics):
        # Check if generation was successful
        if len(raw_text.split()) < TARGET_CHUNK_LENGTH:
            failed_count += 1

            # Don't retry immediately, just skip
            # Retries slow things down significantly
            continue

        # Extract random chunk
        final_chunk = get_random_chunk(raw_text, target_words=TARGET_CHUNK_LENGTH)

        # Validate chunk
        if len(final_chunk.split()) < TARGET_CHUNK_LENGTH * 0.8:  # At least 80% of target
            failed_count += 1
            continue

        # Store formatted data
        entry = {
            "text": final_chunk,
            "label": 1
        }
        dataset.append(entry)

        if len(dataset) >= NUM_SAMPLES:
            break

    # OPTIMIZATION: Clear CUDA cache periodically
    if device == 0 and batch_idx % 50 == 0:
        torch.cuda.empty_cache()

generation_time = time.time() - generation_start

# --- 5. Handle any remaining samples needed ---
# If we didn't get enough samples due to failures, generate more
attempts = 0
max_attempts = 100

while len(dataset) < NUM_SAMPLES and attempts < max_attempts:
    topic = random.choice(TOPICS)
    template = random.choice(TEMPLATES)
    prompt = template.format(topic=topic)

    raw_texts = generate_batch([prompt], max_new_tokens=400)
    raw_text = raw_texts[0]

    if len(raw_text.split()) >= TARGET_CHUNK_LENGTH:
        final_chunk = get_random_chunk(raw_text, target_words=TARGET_CHUNK_LENGTH)

        if len(final_chunk.split()) >= TARGET_CHUNK_LENGTH * 0.8:
            entry = {
                "text": final_chunk,
                "label": 1
            }
            dataset.append(entry)

    attempts += 1

# --- 6. Save Output ---
print("\n" + "=" * 70)
print(f"Generation Complete!")
print(f"  Successful samples: {len(dataset)}")
print(f"  Failed generations: {failed_count}")
print(f"  Success rate: {len(dataset)/(len(dataset)+failed_count)*100:.1f}%")
print(f"  Total time: {generation_time/60:.2f} minutes")
print(f"  Average time per sample: {generation_time/len(dataset):.2f} seconds")

with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"\nData saved to {OUTPUT_FILENAME}")

# Statistics
word_counts = [len(entry['text'].split()) for entry in dataset]
print(f"\nDataset Statistics:")
print(f"  Average words: {sum(word_counts)/len(word_counts):.1f}")
print(f"  Min words: {min(word_counts)}")
print(f"  Max words: {max(word_counts)}")

# Show sample
print(f"\n{'='*70}")
print("Sample output (first 200 chars):")
print(dataset[0]['text'][:200] + "...")

# Clean up
if device == 0:
    torch.cuda.empty_cache()

print("\nDone!")
