# 🏥 MEDI CLEAR AI
# MEDI CLEAR AI — Hybrid RAG for Hospital Billing Verification

# 🏥 MEDI CLEAR AI
### Agentic Hybrid RAG for Insurance and Hospital Billing Verification

MEDI CLEAR AI is an AI-powered insurance and hospital billing verification assistant built using **Hybrid Retrieval-Augmented Generation (RAG)**. It helps users retrieve patient claim information, understand insurance policies, identify missing documents, and obtain concise answers from a structured dataset.

The project combines keyword-based search, semantic search, and AI-generated responses to make insurance and billing information easier to access.

## 🚀 Live Application

- **Live Demo:** https://sarika-845-medi-clear-ai-app-n0xewy.streamlit.app
- **GitHub Repository:** https://github.com/sarika-845/medi-clear-ai

## 📌 Problem Statement

Hospital billing and insurance verification can require staff to check patient records, claim statuses, insurance coverage, exclusions, and required documents manually.

This process can be time-consuming and make it difficult to find the right information quickly.

MEDI CLEAR AI addresses this problem by retrieving relevant patient and insurance information and generating concise responses to natural-language questions.

## 🎯 Project Objectives

- Retrieve patient claim information using natural-language questions.
- Search insurance policies and coverage information.
- Identify pending and rejected claims and their associated document requirements.
- Combine keyword-based and semantic retrieval.
- Generate answers grounded in retrieved information.
- Handle unknown patients without inventing claim details.
- Evaluate answer quality using RAGAS metrics.

## ✨ Key Features

- **Hybrid Retrieval:** Combines BM25 keyword retrieval with Qdrant vector search.
- **Semantic Search:** Uses Sentence Transformers to represent text as embeddings.
- **Insurance Verification:** Retrieves relevant insurance policy information.
- **Claim Status Lookup:** Supports approved, pending, and rejected claim queries.
- **Missing Document Identification:** Retrieves missing-document information from claim records.
- **Agent Routing:** Uses an agent layer to route queries and apply validation logic.
- **Grounded Responses:** Uses retrieved information to generate concise answers.
- **RAGAS Evaluation:** Measures retrieval quality, faithfulness, and answer correctness.
- **Interactive UI:** Provides a simple Streamlit interface.

## 🧠 How It Works

The application follows a retrieval-and-generation workflow.

1. **Load the dataset:** Read patient, claim, billing, and insurance policy records from `insurance_dataset.json`.
2. **Create embeddings:** Use the Sentence Transformers model `all-MiniLM-L6-v2` to represent text as numerical vectors.
3. **Index information:** Store vectors in an in-memory Qdrant collection and prepare a BM25 keyword-search corpus.
4. **Process the question:** Accept a natural-language question through the Streamlit interface.
5. **Retrieve relevant information:** Use keyword and semantic retrieval to find relevant records.
6. **Route and validate:** Apply the agent layer's routing and validation logic where appropriate.
7. **Generate the response:** Use the retrieved context to produce a concise answer.
8. **Display the result:** Return the response in the Streamlit application.

## 🏗️ Architecture

```text
              User Question
                    |
                    v
             Streamlit UI
                 app.py
                    |
                    v
          Agent Routing & Validation
                 agent.py
                    |
                    v
             Hybrid Retrieval
                RAG.py
                    |
           +--------+--------+
           |                 |
           v                 v
     BM25 Retrieval     Qdrant Search
     Keyword Search     Semantic Search
           |                 |
           +--------+--------+
                    |
                    v
          Relevant Information
                    |
                    v
          LLM Answer Generation
                    |
                    v
            Final Response
                    |
                    v
             Streamlit UI
```

*The diagram represents the project's conceptual workflow. The exact execution order and components depend on the implementation in the source files.*

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive web interface |
| Sentence Transformers | Text embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| Qdrant | Vector search |
| BM25 (`rank_bm25`) | Keyword-based retrieval |
| Hybrid RAG | Combines lexical and semantic retrieval |
| Hugging Face Transformers | Language-model support |
| RAGAS | RAG evaluation |
| JSON | Patient, claim, and policy data |
| Git and GitHub | Version control and source hosting |

## 📂 Project Structure

```text
medi_clear_rag_app/
│
├── app.py                  # Streamlit user interface
├── agent.py                # Agent routing and validation
├── RAG.py                  # Hybrid retrieval and generation
├── insurance_dataset.json  # Patient, claim and policy records
├── evaluate_rag.py         # RAGAS evaluation script
├── ragas_results.json      # Saved evaluation results
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git exclusions
└── venv/                   # Local Python environment
```

The `venv/` folder is for local development and should not be committed to GitHub.

## 📊 Dataset

The demonstration uses a synthetic, single-hospital insurance billing dataset.

| Dataset component | Records |
|---|---:|
| Patient records | 24 |
| Insurance policies | 10 |
| Total records across these two groups | 34 |

The patient records include claim information such as:

- Claim ID and status
- Claim processing stage
- Pre-authorization status
- Document verification status
- Missing documents
- Rejection reasons
- Submission and approval dates
- Insurance-covered and patient-payable amounts

The insurance policy records contain information such as coverage, exclusions, claim timelines, and required documents.

**Note:** This is a demonstration dataset, not a production hospital database.

## 🔎 Example Questions

Users can ask questions such as:

**Patient and claim queries**
- What is the claim status for Arjun Malhotra?
- What is the claim status for Rahul Sethi?
- What documents are missing for Vivek Anand?
- Why was Pooja Shah's claim rejected?

**Insurance policy queries**
- What documents are required for a cardiac procedure claim?
- Which treatments are covered under a policy?
- What exclusions apply to the insurance policy?

**Unknown patient handling**
- What is the claim status for Alex Unknown?

When a patient cannot be found, the application should indicate that the record is unavailable instead of inventing a claim status.

## 🧪 RAGAS Evaluation

RAGAS (Retrieval-Augmented Generation Assessment) is used to evaluate the quality of the RAG pipeline.

The project evaluates four metrics.

| Metric | What it measures |
|---|---|
| Context Precision | Whether retrieved context is relevant to the question |
| Context Recall | Whether the retrieved context contains the information needed to answer the question |
| Faithfulness | Whether the generated answer is supported by the retrieved context |
| Answer Correctness | How closely the generated answer matches the reference answer |

### Initial Evaluation Results

The initial evaluation used five patient claim-status questions.

| Metric | Recorded result |
|---|---:|
| Context Precision | Approximately 1.000 |
| Context Recall | 1.000 |
| Faithfulness | 1.000 |
| Average Answer Correctness | 0.9436 (94.36%) |

**Evaluation scope:** Five questions covering the claim status of five patients.

The average Answer Correctness is calculated from the five saved scores, including the final score for Rohan Kapoor.

These values represent the initial evaluation dataset and should not be interpreted as guaranteed performance on unseen questions.

### Functional Testing

Additional manual tests were performed through the deployed application.

| Test case | Observed result |
|---|---|
| Rejected claim — Pooja Shah | Identified the missing pre-authorization confirmation |
| Unknown patient — Alex Unknown | Reported that the patient could not be found |

Both observed responses matched the expected behavior for these test cases.

These manual tests are separate from the five-question RAGAS evaluation. Their results have not been added to the RAGAS metrics.

### Evaluation Files

- `evaluate_rag.py` — runs the RAGAS evaluation workflow.
- `ragas_results.json` — stores the saved evaluation scores.

The evaluation workflow uses an LLM and embeddings for metric calculation. API availability, quota limits, and model compatibility can affect whether an evaluation run completes.

## ⚙️ Local Installation

### Prerequisites

- Python installed on your system
- Git
- A terminal such as PowerShell
- Required model files or internet access for downloading models

### 1. Clone the repository

```bash
git clone https://github.com/sarika-845/medi-clear-ai.git
cd medi-clear-ai
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

### 3. Install dependencies

pip install -r requirements.txt
### 4. Run the application
streamlit run app.py

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

*Run these commands from the directory containing `app.py` and `requirements.txt`. If your cloned repository has a different folder structure, navigate to the appropriate directory first.*

## 🔐 Configuration and Security

- Do not commit API keys, tokens, passwords, or private credentials.
- Store required API keys in environment variables or the deployment platform's secrets manager.
- Keep `venv/` and Python cache files out of version control.
- Use synthetic or properly anonymized data for demonstrations.
- Do not use this demonstration as the sole basis for real insurance or medical billing decisions.

## ⚠️ Current Limitations

- The current evaluation set contains only five questions.
- Evaluation scores may not generalize to unseen questions.
- Qdrant is configured for in-memory demonstration use; its index is not persistent across application restarts.
- The application depends on the completeness and correctness of the supplied dataset.
- Model availability, latency, API quotas, and dependency versions may affect performance.
- Further testing is needed before considering the application production-ready.



## ⚠️ Challenges Faced

- **Retrieval Quality:** Combining BM25 and Qdrant for relevant results.
- **LLM Response Quality:** Improving answer accuracy and relevance.
- **API Rate Limits:** Switching to Groq when Gemini quota was exhausted.
- **RAGAS Integration:** Resolving embedding compatibility and evaluating answer quality.
- **Performance:** Managing model-loading time and response latency.

## 🔮 Future Improvements

- Expand RAGAS evaluation with more test cases.
- Improve retrieval accuracy and answer generation.
- Add persistent Qdrant storage.
- Integrate Langfuse for tracing and monitoring.
- Optimize response time and resource usage.
- Strengthen testing, security, and data privacy.



*MEDI CLEAR AI is an educational and demonstration project designed to explore Hybrid RAG, AI-assisted insurance verification, and RAG evaluation.*

