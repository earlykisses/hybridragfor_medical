# FaithfulMed

## Evidence-Grounded Medical Question Answering and Claim Extraction

FaithfulMed is a medical Retrieval-Augmented Generation (RAG) system designed to generate evidence-grounded answers and explanations for medical questions.

The project combines:

* Dense Retrieval (PubMedBERT)
* Sparse Retrieval (BM25)
* Hybrid Retrieval (Reciprocal Rank Fusion)
* Medical LLM Generation (Qwen 2.5)
* Claim Extraction

The goal is to ensure that generated medical responses are supported by retrieved medical evidence before further hallucination verification modules are introduced.

---

# Features

## Hybrid Medical Retrieval

FaithfulMed retrieves evidence from multiple medical knowledge sources using:

### Dense Retrieval

Model:

* S-PubMedBERT-MS-MARCO

Advantages:

* Semantic understanding
* Handles paraphrased medical queries
* Strong biomedical representation

---

### Sparse Retrieval

Model:

* BM25

Advantages:

* Exact keyword matching
* Fast retrieval
* Strong lexical recall

---

### Hybrid Retrieval

Method:

* Reciprocal Rank Fusion (RRF)

Combines:

* Dense Retrieval
* BM25 Retrieval

Benefits:

* Better recall
* More robust evidence retrieval
* Reduced retrieval failures

---

## Medical Answer Generation

Generator:

* Qwen 2.5 7B Instruct
* Local inference using Ollama

Input:

* User Question
* Retrieved Medical Evidence

Output:

* Final Answer
* Detailed Explanation

Example:

Question:

Does mitochondria play a role in programmed cell death?

Answer:

Yes.

Explanation:

Mitochondrial dynamics were shown to correlate with programmed cell death progression in the lace plant. Experimental evidence demonstrated that mitochondrial changes occurred during different stages of cell death and that cyclosporine A treatment altered these dynamics.

---

## Claim Extraction

The generated explanation is decomposed into atomic claims.

Example:

Explanation:

"Mitochondrial dynamics are involved in programmed cell death. Cyclosporine A treatment reduced perforation formation."

Extracted Claims:

1. Mitochondrial dynamics are involved in programmed cell death.
2. Cyclosporine A treatment reduced perforation formation.

These claims will later be used for evidence verification and hallucination detection.

---

# Architecture

Medical Question
↓
Hybrid Retriever
(Dense + BM25 + RRF)
↓
Retrieved Evidence
↓
Qwen 2.5 Generator
↓
Answer + Explanation
↓
Claim Extractor
↓
Atomic Medical Claims

---

# Medical Knowledge Sources

The retrieval corpus is built from three medical datasets.

| Source           | Passages |
| ---------------- | -------: |
| PubMedQA         |    3,358 |
| MedRAG Textbooks |  125,847 |
| StatPearls       |  262,407 |
| Total            |  391,612 |

---

# Tech Stack

### Retrieval

* ChromaDB
* BM25
* SentenceTransformers
* PubMedBERT

### Generation

* Ollama
* Qwen 2.5 7B Instruct

### Data Processing

* Python
* Hugging Face Datasets

---

# Project Structure

```text
major_project/

├── data/
│   ├── corpus.jsonl
│   ├── textbooks_corpus.jsonl
│   ├── statpearls_corpus.jsonl
│   └── merged_corpus.jsonl
│
├── generation/
│   ├── qwen_generator.py
│   └── claim_extractor.py
│
├── retrieval/
│   ├── dense_retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── bm25.pkl
│   └── bm25_full.pkl
│
├── scripts/
│   ├── build_chroma_full.py
│   ├── build_bm25_full.py
│   ├── evaluate_retrieval_full.py
│   ├── test_hybrid_full.py
│   ├── test_qwen.py
│   ├── test_claim_extractor.py
│   └── test_full_pipeline.py
│
├── vector_db/
├── requirements.txt
├── README.md
└── main.py
```

# Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/FaithfulMed.git

cd FaithfulMed
```

Create virtual environment:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Ollama Setup

Install Ollama:

https://ollama.com

Download Qwen:

```bash
ollama pull qwen2.5:7b
```

Verify:

```bash
ollama list
```

Expected:

```text
qwen2.5:7b
```

---

# Running the Pipeline

## Test Retrieval

```bash
python scripts/test_hybrid_full.py
```

---

## Test Qwen Generation

```bash
python scripts/test_qwen.py
```

---

## Test Claim Extraction

```bash
python scripts/test_claim_extractor.py
```

---

## Run Complete Pipeline

```bash
python scripts/test_full_pipeline.py
```

Pipeline:

Question
→ Hybrid Retrieval
→ Evidence Retrieval
→ Qwen Generation
→ Claim Extraction

---

# Example Output

Question:

Does mitochondria play a role in programmed cell death?

Answer:

Yes.

Explanation:

Experimental evidence demonstrated that mitochondrial dynamics correlate with programmed cell death progression and that cyclosporine A treatment influences these processes.

Extracted Claims:

1. Mitochondrial dynamics are involved in programmed cell death.
2. Cyclosporine A treatment affects programmed cell death progression.

---

# Current Progress

## Completed

* PubMedQA Corpus Construction
* Textbooks Corpus Construction
* StatPearls Corpus Construction
* Corpus Merging
* ChromaDB Indexing
* BM25 Indexing
* Dense Retrieval
* BM25 Retrieval
* Hybrid Retrieval
* Reciprocal Rank Fusion
* Qwen Integration
* Explanation Generation
* Claim Extraction
* End-to-End Pipeline

## Next Steps

* Retrieval Evaluation
* Dense vs BM25 vs Hybrid Comparison
* Claim Extractor V2
* Evidence Verification Module
* Hallucination Detection Framework

---

# License

MIT License
