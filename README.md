# 🏥 MEDI CLEAR AI

### Hybrid RAG-Based Hospital Insurance & Billing Assistant

MEDI CLEAR AI is a Generative AI application that helps users retrieve and understand hospital insurance and billing information using natural-language questions.

The system combines **BM25 keyword retrieval** and **Qdrant semantic vector search**, uses **Reciprocal Rank Fusion (RRF)** to combine the rankings, and passes the most relevant context to **Qwen2.5-3B-Instruct** to generate a concise answer.

---

## 🎯 Problem Statement

Hospitals handle large amounts of patient billing and insurance information, including:

- Patient billing details
- Claim status
- Claim stages
- Insurance-covered amounts
- Patient payable amounts
- Missing documents
- Policy coverage
- Deductibles
- Copay
- Exclusions
- Required documents
- Claim timelines

Finding the required information manually can be time-consuming.

MEDI CLEAR AI provides a natural-language interface where users can simply ask a question and receive a relevant answer.

---

## 💡 Objective

The main objective of MEDI CLEAR AI is to:

- Reduce manual searching of hospital insurance records
- Retrieve relevant patient and policy information
- Answer questions using natural language
- Reduce hallucination by grounding answers in retrieved records
- Demonstrate a practical Generative AI + RAG use case

---

## 🧠 System Architecture

```text
                    USER QUESTION
                         │
                         ▼
                 QUERY CLASSIFICATION
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       PATIENT / POLICY          GENERAL
              │                     │
              ▼                     ▼
       ┌───────────────┐        Qwen2.5-3B
       │ Hybrid Search │        Direct Answer
       └───────┬───────┘
               │
       ┌───────┴────────┐
       ▼                ▼
     BM25             Qdrant
   Keyword          Semantic
   Search            Search
       │                │
       └───────┬────────┘
               ▼
             RRF
    Reciprocal Rank Fusion
               │
               ▼
             TOP 1
               │
               ▼
       Qwen2.5-3B-Instruct
               │
               ▼
          FINAL ANSWER





## 🔎 Hybrid RAG

MEDI CLEAR AI uses a hybrid retrieval approach.

BM25

BM25 performs keyword-based retrieval.

It is useful for exact terms such as:

Patient names
Patient IDs
Policy IDs
Claim IDs
Insurance terminology
Qdrant

Qdrant is used as the vector database.

The project uses the:

all-MiniLM-L6-v2

Sentence Transformer model to generate vector embeddings.

This allows the system to retrieve information based on semantic similarity rather than only exact keyword matches.

Reciprocal Rank Fusion

The BM25 and Qdrant rankings are combined using Reciprocal Rank Fusion (RRF).

The highest-ranked document after fusion is selected as the Top-1 context.

## 🤖 Large Language Model

The project uses:

Qwen/Qwen2.5-3B-Instruct

The model generates the final natural-language response using the retrieved context.

For patient and policy questions, the model is instructed to:

Use only the retrieved information
Avoid inventing information
Provide a short and direct response

For general insurance questions, the system directly uses Qwen without retrieving patient records.


## 🔄 RAG Workflow
User Question
      ↓
Query Classification
      ↓
Patient / Policy / General
      ↓
Patient or Policy
      ↓
BM25 + Qdrant
      ↓
RRF Ranking
      ↓
Top-1 Document
      ↓
Qwen2.5-3B-Instruct
      ↓
Clean Answer

## 🗂️ Dataset

The project uses a JSON-based dataset containing:

Patient Information
Patient ID
Patient name
Treatment
Policy ID
Total bill
Claim status
Claim stage
Insurance-covered amount
Patient payable amount
Missing documents
Policy Information
Policy ID
Policy name
Coverage limit
Reimbursement percentage
Deductible
Copay
Room rent limit
Covered treatments
Exclusions
Required documents
Claim timeline

The dataset is intended for demonstration and development purposes.

# 💬 Example Questions
Patient Queries
What is the claim status for Priya Sharma?

What is Priya Sharma's total bill?

How much does Priya Sharma need to pay?

How much of Priya Sharma's bill is covered by insurance?

Are any documents missing for Priya Sharma's claim?
Policy Queries
What is the deductible for POL002?

What is the coverage limit for POL002?

What is the copay percentage for POL002?

What treatments are covered under POL002?

What are the exclusions under POL002?

What documents are required for maternity claims?
General Questions
What is a health insurance deductible?

What is copay in health insurance?

What does claim pending mean?

What is reimbursement in health insurance?
## 🛠️ Technologies Used
Technology	Purpose
Python	Core development
Streamlit	Web application
Qwen2.5-3B-Instruct	Answer generation
Hugging Face Transformers	LLM implementation
Sentence Transformers	Text embeddings
all-MiniLM-L6-v2	Embedding model
Qdrant	Vector database
BM25	Keyword retrieval
RRF	Hybrid ranking
PyTorch	Model inference
JSON	Dataset storage
📁 Project Structure
medi-clear-ai/
│
├── app.py
├── RAG.py
├── insurance_dataset.json
├── requirements.txt
├── README.md
├── .gitignore
│
└── venv/              # Not uploaded to GitHub
⚙️ Installation

Clone the repository:

git clone https://github.com/sarika-845/medi-clear-ai.git

Navigate into the project:

cd medi-clear-ai

Create a virtual environment:

python -m venv venv

Activate the environment on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application:

python -m streamlit run app.py

The application will open in your browser.

🧪 Testing

The system can be tested using:

Patient-specific questions
Policy-specific questions
General insurance questions
Unknown patient/policy queries

The application is designed to avoid generating patient-specific information when it is not present in the available records.

## 🚀 Future Improvements

Possible future improvements include:

PDF insurance document ingestion
Automatic document chunking
Larger production-scale datasets
Conversation memory
Authentication and role-based access
Hospital staff dashboard
Claim document verification
Advanced RAG evaluation
Response confidence scoring
Cloud deployment
GPU-based inference optimization
👨‍💻 Project Highlights

This project demonstrates practical implementation of:

Generative AI
Retrieval-Augmented Generation (RAG)
Hybrid Search
BM25 Retrieval
Vector Search
Qdrant
Sentence Transformers
Reciprocal Rank Fusion
LLM-based Answer Generation
Streamlit Application Development
