# 🏥 MEDI CLEAR AI
# MEDI CLEAR AI — Hybrid RAG for Hospital Billing Verification

## 1. Problem Statement

Hospital billing teams must check patient records, insurance policies, and claim information before answering billing-related questions. Manually searching through these records takes time and can lead to errors.

Traditional chatbots may also generate incorrect information when the required details are unavailable.

## 2. Objective

MEDI CLEAR AI is an AI-powered application that helps users retrieve patient claim information and insurance policy details through natural-language questions.

The objectives are to:
- Retrieve relevant information from a structured insurance dataset.
- Combine keyword-based and semantic search for better retrieval.
- Generate concise answers grounded in retrieved information.
- Reduce unsupported answers by validating patient and policy information.
- Evaluate answer quality using RAGAS metrics.

**Important:** This is a demonstration project using sample data, not a production hospital billing system.

## 3. How the Project Works

```text
User asks a question
        ↓
    Streamlit UI
        ↓
      agent.py
        ↓
 Classify the question
        ↓
 ┌───────────────┬───────────────┐
 │ Patient/Policy│ General Query │
 ↓               ↓
 Validate        Direct LLM
 required data   response
 ↓
 Hybrid Search
 BM25 + Qdrant
 ↓
 RRF Ranking
 ↓
 Retrieve Top 1
 relevant document
 ↓
 Qwen2.5 LLM
 ↓
 Clean answer
        ↓
 Display in Streamlit
```

RAGAS evaluation is a separate process. It measures answer quality using test questions, reference answers, retrieved context, and generated answers.

## 4. Project Files and Why We Use Them

| File | Purpose |
|---|---|
| `app.py` | Provides the Streamlit user interface where users enter questions and view answers. |
| `agent.py` | Classifies questions, validates patient information, and routes requests to the appropriate workflow. |
| `RAG.py` | Implements hybrid retrieval, document ranking, and LLM-based answer generation. |
| `insurance_dataset.json` | Stores the sample patient, claim, and insurance policy information used by the application. |
| `evaluate_rag.py` | Runs the RAGAS evaluation on selected test cases. |
| `ragas_results.json` | Stores completed evaluation scores so the evaluation can resume without discarding previous results. |
| `requirements.txt` | Lists the Python packages required to run the project. |
| `.gitignore` | Prevents unnecessary files, such as virtual environments and local secrets, from being committed to GitHub. |

## 5. Technologies Used

| Technology | Why it is used |
|---|---|
| Python | Implements the application logic. |
| Streamlit | Creates the interactive web interface. |
| BM25 | Finds documents using exact words and keyword relevance. |
| Qdrant | Stores and searches vector representations of documents. |
| Sentence Transformers | Converts text into numerical embeddings for semantic search. |
| Reciprocal Rank Fusion (RRF) | Combines the rankings from BM25 and Qdrant into one ranked list. |
| Qwen2.5-0.5B-Instruct | Generates natural-language answers using the retrieved context. |
| RAGAS | Evaluates retrieval quality and answer quality. |
| GitHub and Streamlit Cloud | Store the source code and host the web application. |

## 6. Why Hybrid RAG?

A single search method has limitations.

- **BM25 — Keyword search:** Works well when the question contains an exact patient name, ID, claim term, or policy keyword.
- **Qdrant — Semantic search:** Finds conceptually related text even when the wording differs.
- **RRF — Rank fusion:** Combines the two search rankings without requiring their scores to use the same scale.
- **Top-1 retrieval:** Passes the highest-ranked document to the answer generator to keep the context focused.

The retrieved document provides evidence for the LLM's response instead of relying only on its pretrained knowledge.

## 7. Role of the Agent

The agent in `agent.py` decides how a question should be handled.

1. **Patient query:** Checks whether the patient's name or ID exists in the dataset before retrieving claim information.
2. **Policy query:** Searches the available policy information.
3. **General query:** Routes the question to the LLM without the patient/policy retrieval workflow.

These checks help prevent the application from inventing records for unknown patients. They do not guarantee that every generated answer is correct.

## 8. RAGAS Evaluation

RAGAS evaluates different aspects of the RAG pipeline.

| Metric | What it measures |
|---|---|
| Context Precision | Whether the retrieved context contains relevant information near the top of the results. |
| Context Recall | Whether the retrieved context contains the information needed for the reference answer. |
| Faithfulness | Whether the generated answer is supported by the retrieved context. |
| Answer Correctness | How closely the generated answer matches the reference answer in meaning and factual content. |

### Current evaluation status

The current evaluation uses five selected patient test cases. The last recorded results were:

| Metric | Average score | Completed scores |
|---|---:|---:|
| Context Precision | 1.000 | 5/5 |
| Context Recall | 1.000 | 5/5 |
| Faithfulness | 1.000 | 5/5 |
| Answer Correctness | 0.930 | 4/5 |

**Status:** The evaluation is not yet complete. One Answer Correctness score was pending because the Gemini API daily quota was reached. These results are preliminary and apply only to the selected test cases; they do not establish performance across all 24 patients.

## 9. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/sarika-845/medi-clear-ai.git
cd medi-clear-ai
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL to open the application in your browser.

**For RAGAS evaluation:** Configure the required Gemini API key as an environment variable named `GOOGLE_API_KEY`, then run:

```bash
python evaluate_rag.py
```

Keep API keys private. Never commit them to GitHub.

## 10. Example Questions

- What is the claim status for Aarav Mehta?
- What is the claim status for patient ID 102?
- What does the available insurance policy cover?
- Can you explain how hybrid retrieval works in this application?

Use names, IDs, and policies that actually exist in the sample dataset when testing factual questions.

## 11. Limitations and Future Improvements

- Expand the evaluation beyond the five selected patients.
- Complete the pending RAGAS score and report the final results.
- Improve handling of ambiguous questions and multiple matching records.
- Test retrieval and generation on more diverse questions.
- Add monitoring, audit logging, access control, and privacy safeguards before any real-world healthcare use.

## 12. Project Summary

MEDI CLEAR AI demonstrates how an agent, hybrid retrieval, a language model, and RAGAS evaluation can work together to answer questions about sample hospital billing and insurance data.

The project combines information retrieval with language generation and includes validation safeguards to reduce unsupported patient-specific answers.

**Live application:** https://sarika-845-medi-clear-ai-app-a88k7p.streamlit.app

**GitHub repository:** https://github.com/sarika-845/medi-clear-ai
