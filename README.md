# AI Text Detector (DeBERTa-v3)
### Evaluating the Generalization Limits of Transformer-Based Detectors

## 1. Overview
This project investigates the "arms race" between AI text generators and detectors. I engineered a robust detection system fine-tuned on **DeBERTa-v3**, capable of distinguishing human-written Wikipedia articles from AI-generated text with **99.4% accuracy**.

Crucially, this project highlights a significant **"Generalization Gap"** in current NLP research: while the model achieves near-perfect detection on the architecture it was trained on (GPT-Neo), it struggles to generalize to State-of-the-Art (SOTA) models like GPT-4, proving that "universal" detection is an elusive goal.

## 2. Key Features
* **Domain-Aligned Synthetic Data:** Generated 3,000+ "Wikipedia-style" samples using **GPT-Neo-1.3B** to ensure the model learns semantic features, not just topic bias.
* **Leakage Prevention Pipeline:** Implemented a **Random Chunking** strategy to eliminate structural artifacts (e.g., "start-of-sentence" bias) that caused early models to overfit.
* **SOTA Architecture:** Fine-tuned `microsoft/deberta-v3-base` using Hugging Face Trainers with early stopping.
* **Real-World Testing:** Demonstrated the model's high sensitivity to **Domain Shift** (Wikipedia vs. Reddit) and **Model Shift** (GPT-Neo vs. GPT-4).

---

## 3. The "Generalization Gap" Discovery
Unlike standard "toy" projects, this research exposed a critical limitation in AI detection.

| Test Data Source | Model Confidence | Verdict |
| :--- | :--- | :--- |
| **GPT-Neo (In-Distribution)** | **99.9%** | **Detected** |
| **Human Wikipedia** | **74.6%** | **Detected** |
| **GPT-4 (Out-of-Distribution)** | **66.1%** | **Uncertain** |

**Insight:** Detectors trained on legacy models (2021 era) cannot reliably detect modern SOTA models (2024 era) due to the closing gap in perplexity and burstiness.

---

## 4. Methodology
### 1. Data Engineering
* **Human Data:** Scraped 2,000 articles from Wikipedia using the Wikipedia API.
* **AI Data:** Generated 1,600 samples using `EleutherAI/gpt-neo-1.3B`.
* **Preprocessing:** Both datasets were generated at ~600 words and randomly sliced to ~300 words to standardize sentence fragmentation.

### 2. Model Training
* **Base Model:** DeBERTa-v3 (Decoding-enhanced BERT with disentangled attention).
* **Hyperparameters:**
    * Learning Rate: `5e-6`
    * Batch Size: `8`
    * Epochs: `3` (Early Stopping at Epoch 2)

---

## 5. Usage

#### model is live at: https://huggingface.co/vraj33/ai-text-detector-deberta

### 3. Run Inference (Python)
You can use the model directly from Hugging Face without training it yourself:
```
from transformers import pipeline

# Load my fine-tuned model
model_id = "vraj33/wiki-ai-detector-deberta"
classifier = pipeline("text-classification", model=model_id)

# Test text
text = "The quick brown fox jumps over the lazy dog."
result = classifier(text)

print(result)
# Output: [{'label': 'LABEL_0', 'score': 0.99...}] (LABEL_0 = Human, LABEL_1 = AI)
```
