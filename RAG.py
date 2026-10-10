import json
import re
import torch

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- DATA ----------
with open("insurance_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

patients = data["patients"]
policies = data["policies"]

documents = []
types = []

for p in patients:
    c = p["claim"]
    b = p["billing_details"]
    o = p["claim_outcome"]

    documents.append(
        f"Patient ID: {p['patient_id']}\n"
        f"Patient Name: {p['name']}\n"
        f"Treatment: {p['diagnosis_treatment_type']}\n"
        f"Policy ID: {p['policy_id']}\n"
        f"Total Bill: ₹{b['total_bill_amount']}\n"
        f"Claim Status: {c['claim_status']}\n"
        f"Claim Stage: {c['claim_stage']}\n"
        f"Insurance Covered: ₹{o['insurance_covered_amount']}\n"
        f"Patient Payable: ₹{o['patient_payable_amount']}\n"
        f"Missing Documents: {', '.join(c['missing_documents']) or 'None'}"
    )
    types.append("patient")

for p in policies:
    documents.append(
        f"Policy ID: {p['policy_id']}\n"
        f"Policy Name: {p['policy_name']}\n"
        f"Coverage Limit: ₹{p['coverage_limit']}\n"
        f"Reimbursement: {p['reimbursement_percentage']}%\n"
        f"Deductible: ₹{p['deductible']}\n"
        f"Copay: {p['copay_percentage']}%\n"
        f"Room Rent Limit: ₹{p['room_rent_limit_per_day']}\n"
        f"Covered Treatments: {', '.join(p['covered_treatment_categories'])}\n"
        f"Exclusions: {', '.join(p['exclusions'])}\n"
        f"Required Documents: {', '.join(p['required_documents'])}\n"
        f"Claim Timeline: {p['claim_timeline']}"
    )
    types.append("policy")


# ---------- BM25 ----------
def tokenize(text):
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokenize(x) for x in documents])


# ---------- QDRANT ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(":memory:")

qdrant.create_collection(
    collection_name="insurance",
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    )
)

vectors = embedder.encode(
    documents,
    normalize_embeddings=True
)

qdrant.upsert(
    collection_name="insurance",
    points=[
        models.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"text": documents[i], "type": types[i]}
        )
        for i in range(len(documents))
    ]
)


# ---------- CLASSIFIER ----------
def classify_query(query):
    q = query.lower()

    for p in patients:
        if p["name"].lower() in q:
            return "patient"

    if re.search(r"\bPAT\d+\b|\bCLM\d+\b", query, re.I):
        return "patient"

    if re.search(r"\bPOL\d+\b", query, re.I):
        return "policy"

    patient_words = [
        "patient", "claim status", "claim stage",
        "total bill", "bill amount", "billing",
        "patient payable", "insurance covered",
        "missing documents"
    ]

    policy_words = [
        "policy", "coverage", "deductible", "copay",
        "reimbursement", "room rent", "exclusion",
        "required documents", "covered treatment",
        "claim timeline"
    ]

    if any(x in q for x in patient_words):
        return "patient"

    if any(x in q for x in policy_words):
        return "policy"

    return "general"


# ---------- HYBRID RAG ----------
def hybrid_search(query):
    query_type = classify_query(query)

    if query_type == "general":
        return None

    allowed = [
        i for i, t in enumerate(types)
        if t == query_type
    ]

    # Exact ID filtering
    ids = re.findall(r"\b(?:PAT|CLM|POL)\d+\b", query.upper())

    if ids:
        exact = [
            i for i in allowed
            if any(x in documents[i] for x in ids)
        ]
        if exact:
            allowed = exact

    # BM25
    scores = bm25.get_scores(tokenize(query))
    bm25_ids = sorted(
        allowed,
        key=lambda i: scores[i],
        reverse=True
    )[:5]

    # Qdrant
    vector = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name="insurance",
        query=vector,
        limit=5
    ).points

    qdrant_ids = [
        int(r.id)
        for r in results
        if int(r.id) in allowed
    ]

    # RRF
    rrf = {}

    for rank, i in enumerate(bm25_ids, 1):
        rrf[i] = rrf.get(i, 0) + 1 / (60 + rank)

    for rank, i in enumerate(qdrant_ids, 1):
        rrf[i] = rrf.get(i, 0) + 1 / (60 + rank)

    if not rrf:
        return None

    best = max(rrf, key=rrf.get)

    return documents[best]


# ---------- LLM ----------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

llm.eval()


# ---------- GENERATION ----------
def generate_answer(query, context=None):

    if context:
        prompt = f"""
You are MEDI CLEAR AI.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer briefly and directly.
Do not invent information.
"""
    else:
        prompt = f"""
You are MEDI CLEAR AI, a hospital insurance assistant.

Question:
{query}

Give a short and simple explanation.
"""

    messages = [
        {"role": "system", "content": "You are MEDI CLEAR AI."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)

    with torch.inference_mode():
        output = llm.generate(
            input_ids=input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        output[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return answer


# ---------- MAIN ----------
def rag_query(query):
    query_type = classify_query(query)

    if query_type in ["patient", "policy"]:
        context = hybrid_search(query)

        if not context:
            return "I could not find the requested information."

        print("Agent → Hybrid RAG → BM25 + Qdrant → RRF → TOP 1")
        return generate_answer(query, context)

    print("Agent → General LLM")
    return generate_answer(query)