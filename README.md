# 🩺 Perspective-driven Healthcare Summarization

**Efficient Summarization of Healthcare Responses using FLAN-T5, BART, Pegasus, and Weak Supervision**

This project presents a perspective-aware abstractive summarization system for medical Q&A forums (like r/AskDocs, r/Medical_Advice). It generates summaries by extracting and synthesizing user responses from multiple perspectives such as **Information**, **Cause**, **Suggestion**, **Experience**, and **Question**.

---

## 🔗 Code Repository

The complete source code is available at:

👉 **GitHub Repo:** [https://github.com/pandillapelly22345/Perspective-driven-Healthcare-Summarization](https://github.com/pandillapelly22345/Efficient-Summarization-of-Healthcare-Responses---NLP-Project)

---

## 🧠 Motivation

Online health forums provide valuable information, but user-generated answers vary in quality and reliability. Users often struggle to extract meaningful insights due to:
- Redundant or irrelevant answers
- Lack of medical verification
- Diverse response styles (subjective vs. factual)

### ✅ Goal:
Build a summarization system that:
- Understands and classifies responses into perspectives
- Generates coherent, concise, and contextually accurate summaries
- Uses ensemble learning for improved quality

---

## 📊 Dataset

We use the [PUMAAA dataset](https://arxiv.org/abs/2406.08881), consisting of:
- 3 JSON files: `train.json`, `valid.json`, and `test.json`
- Fields include `question`, `context`, `answers`, and `labelled_answer_spans` per perspective
- 5 perspectives: **Information**, **Cause**, **Suggestion**, **Experience**, and **Question**

---

## 🛠️ Methodology

### 🔹 1. Fine-Tuning & Ensemble Learning

- **Flan-T5 + LoRA**: Instruction-based fine-tuning with perspective prompts
- **BART**: Fine-tuned for fluency and coherence
- **Pegasus**: Used for abstractive post-processing in some setups

We compare outputs from BART and Flan-T5 and **stack them using a selection heuristic** (e.g., longer summary or better semantic alignment).

### 🔹 2. Weak Supervision Pipeline

- **Snorkel + Logistic Regression**: Uses keyword LFs to assign pseudo-labels for perspectives
- **Sentence Embeddings + SVM**: Trained to classify sentence spans into perspectives
- **Zero-shot (facebook/bart-large-mnli)**: Backup classification for ambiguous or abstained cases

---

## 🧪 Evaluation
As the summerization was seen better in ensemble method as compared to weak supervision so we used the ensemble method models in the app.

Metrics used:
- **BERTScore** (semantic similarity)
- **BLEU** (token-level overlap)
- **METEOR**

| Model       | BERTScore F1 | BLEU | METEOR |
|-------------|---------------|------|--------|
| BART        | 0.8907        | 0.0883 | 0.2544 |
| Flan-T5     | 0.8632        | 0.0363 | 0.1856 |
| **Stacked** | 0.8815        | 0.0747 | 0.2386 |

---

## 🚀 Deployment

### 🔹 Hugging Face Spaces:
The app is live at: [https://huggingface.co/spaces/harshvardhini123/healthcare-summary](https://huggingface.co/spaces/harshvardhini123/healthcare-summary)

You can enter a medical question and get summaries for the selected perspective.

### 🔹 Hosting on Hugging Face:
Models are hosted at:
- 🧠 `harshvardhini123/fine_tuned_model` – Flan-T5 + LoRA
- 🧠 `harshvardhini123/fine_tuned_bart` – Fine-tuned BART

These are loaded using:
```python
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
bart_model = BartForConditionalGeneration.from_pretrained("harshvardhini123/fine_tuned_bart")
t5_model = T5ForConditionalGeneration.from_pretrained("harshvardhini123/fine_tuned_model")
