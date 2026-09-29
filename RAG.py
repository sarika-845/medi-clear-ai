import json
import re
import torch

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. LOAD DATA
# ============================================================

with open("insurance_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

patients = data["patients"]
policies = data["policies"]


# ============================================================
# 2. CREATE DOCUMENTS
# ============================================================

documents = []
doc_types = []


# -------------------------
# Patient documents
# -------------------------

for patient in patients:

    claim = patient["claim"]
    billing = patient["billing_details"]
    outcome = patient["claim_outcome"]

    document = f"""
Patient ID: {patient['patient_id']}
Patient Name: {patient['name']}
Treatment: {patient['diagnosis_treatment_type']}
Policy ID: {patient['policy_id']}
Total Bill: ₹{billing['total_bill_amount']}
Claim Status: {claim['claim_status']}
Claim Stage: {claim['claim_stage']}
Insurance Covered: ₹{outcome['insurance_covered_amount']}
Patient Payable: ₹{outcome['patient_payable_amount']}
Missing Documents: {', '.join(claim['missing_documents']) or 'None'}
"""

    documents.append(document)
    doc_types.append("patient")


# -------------------------
# Policy documents
# -------------------------

for policy in policies:

    document = f"""
Policy ID: {policy['policy_id']}
Policy Name: {policy['policy_name']}
Coverage Limit: ₹{policy['coverage_limit']}
Reimbursement: {policy['reimbursement_percentage']}%
Deductible: ₹{policy['deductible']}
Copay: {policy['copay_percentage']}%
Room Rent Limit: ₹{policy['room_rent_limit_per_day']}
Covered Treatments: {', '.join(policy['covered_treatment_categories'])}
Exclusions: {', '.join(policy['exclusions'])}
Required Documents: {', '.join(policy['required_documents'])}
Claim Timeline: {policy['claim_timeline']}
"""

    documents.append(document)
    doc_types.append("policy")


# ============================================================
# 3. BM25
# ============================================================

def tokenize(text):
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi(
    [tokenize(doc) for doc in documents]
)


# ============================================================
# 4. QDRANT
# ============================================================

print("Loading embedding model...")

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


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
            payload={
                "text": documents[i],
                "type": doc_types[i]
            }
        )
        for i in range(len(documents))
    ]
)


print("Hybrid retrieval ready.")


# ============================================================
# 5. QUERY CLASSIFICATION
# ============================================================

def classify_query(query):

    q = query.lower().strip()


    # Patient names
    for patient in patients:

        if patient["name"].lower() in q:
            return "patient"


    # Patient IDs
    if re.search(r"\bPAT\d+\b", query, re.IGNORECASE):
        return "patient"


    # Claim IDs
    if re.search(r"\bCLM\d+\b", query, re.IGNORECASE):
        return "patient"


    # Policy IDs
    if re.search(r"\bPOL\d+\b", query, re.IGNORECASE):
        return "policy"


    # Patient-related questions
    patient_keywords = [
        "patient",
        "claim status",
        "claim for",
        "claim stage",
        "total bill",
        "bill amount",
        "billing status",
        "patient payable",
        "insurance covered",
        "covered amount",
        "missing documents"
    ]

    if any(word in q for word in patient_keywords):
        return "patient"


    # Policy-related questions
    policy_keywords = [
        "policy",
        "coverage",
        "coverage limit",
        "deductible",
        "copay",
        "co-pay",
        "reimbursement",
        "room rent",
        "exclusion",
        "exclusions",
        "required documents",
        "covered treatment",
        "covered treatments",
        "claim timeline"
    ]

    if any(word in q for word in policy_keywords):
        return "policy"


    # General question
    return "general"


# ============================================================
# 6. HYBRID SEARCH
#    BM25 + QDRANT + RRF
# ============================================================

def hybrid_search(query):

    query_type = classify_query(query)


    # General questions do not access records
    if query_type == "general":
        return None


    # Only search relevant document type
    allowed_ids = [
        i
        for i, doc_type in enumerate(doc_types)
        if doc_type == query_type
    ]


    # ========================================================
    # BM25
    # ========================================================

    bm25_scores = bm25.get_scores(
        tokenize(query)
    )


    bm25_results = sorted(
        allowed_ids,
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:5]


    # ========================================================
    # QDRANT
    # ========================================================

    query_vector = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()


    qdrant_results = qdrant.query_points(
        collection_name="insurance",
        query=query_vector,
        limit=5
    ).points


    qdrant_ids = [
        int(result.id)
        for result in qdrant_results
        if int(result.id) in allowed_ids
    ]


    # ========================================================
    # RRF
    # ========================================================

    rrf_scores = {}


    # BM25 ranking
    for rank, index in enumerate(
        bm25_results,
        start=1
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0)
            + 1 / (60 + rank)
        )


    # Qdrant ranking
    for rank, index in enumerate(
        qdrant_ids,
        start=1
    ):

        rrf_scores[index] = (
            rrf_scores.get(index, 0)
            + 1 / (60 + rank)
        )


    if not rrf_scores:
        return None


    # ========================================================
    # TOP 1
    # ========================================================

    best_id = max(
        rrf_scores,
        key=rrf_scores.get
    )


    return documents[best_id]


# ============================================================
# 7. LOAD QWEN 2.5
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


# Explicit device
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("MEDI CLEAR AI DEVICE:", DEVICE)


# ============================================================
# 8. TOKENIZER
# ============================================================

print("Loading Qwen tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 9. MODEL
# ============================================================

print("Loading Qwen2.5-3B-Instruct...")


if DEVICE == "cuda":

    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=None
    )

else:

    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None
    )


# Put model on same device
llm = llm.to(DEVICE)

llm.eval()


print("Qwen model loaded successfully.")


# ============================================================
# 10. GENERATE ANSWER
# ============================================================

def generate_answer(query, context=None):


    # --------------------------------------------------------
    # Patient / Policy
    # --------------------------------------------------------

    if context:

        prompt = f"""
You are MEDI CLEAR AI, a hospital insurance and billing assistant.

Use ONLY the information provided below.

Context:
{context}

Question:
{query}

Give a short and direct answer.

Do not invent information.
Do not add facts that are not present in the context.
"""


    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    else:

        prompt = f"""
You are MEDI CLEAR AI, a hospital insurance assistant.

Question:
{query}

Give a simple, accurate and concise explanation.
"""


    # ========================================================
    # CHAT TEMPLATE
    # ========================================================

    messages = [

        {
            "role": "system",
            "content": (
                "You are MEDI CLEAR AI. "
                "Provide accurate and concise "
                "hospital insurance information."
            )
        },

        {
            "role": "user",
            "content": prompt
        }

    ]


    # ========================================================
    # IMPORTANT FIX
    # ========================================================

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )


    # Get ONLY input_ids
    input_ids = inputs["input_ids"].to(DEVICE)


    # ========================================================
    # GENERATE
    # ========================================================

    with torch.inference_mode():

        outputs = llm.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )


    # ========================================================
    # REMOVE PROMPT
    # ========================================================

    generated_tokens = outputs[
        0,
        input_ids.shape[-1]:
    ]


    # ========================================================
    # DECODE
    # ========================================================

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    return answer


# ============================================================
# 11. CLEAN ANSWER
# ============================================================

def clean_answer(answer):

    answer = answer.strip()


    unwanted_prefixes = [
        "Answer:",
        "Response:",
        "Assistant:"
    ]


    for prefix in unwanted_prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()


    return answer


# ============================================================
# 12. MAIN RAG PIPELINE
# ============================================================

def rag_query(query):

    query = query.strip()


    if not query:

        return "Please enter a question."


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    query_type = classify_query(query)


    print()
    print("Query:", query)
    print("Query Type:", query_type)


    # ========================================================
    # PATIENT / POLICY
    # ========================================================

    if query_type in [
        "patient",
        "policy"
    ]:

        context = hybrid_search(query)


        if context is None:

            return (
                "I could not find the requested "
                "information in the records."
            )


        print("Retrieval: BM25 + Qdrant")
        print("Ranking: RRF")
        print("Selected: TOP 1")


        answer = generate_answer(
            query,
            context
        )


    # ========================================================
    # GENERAL
    # ========================================================

    else:

        print("Retrieval: None")
        print("Generation: Qwen2.5-3B")


        answer = generate_answer(
            query
        )


    # ========================================================
    # CLEAN
    # ========================================================

    return clean_answer(answer)


# ============================================================
# 13. TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MEDI CLEAR AI")
    print("Hybrid RAG Insurance & Billing Assistant")
    print("=" * 60)
    print()


    question = input(
        "Ask MEDI CLEAR AI: "
    )


    answer = rag_query(
        question
    )


    print()
    print("Response:")
    print(answer)