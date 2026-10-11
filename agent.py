import json
import re
from pathlib import Path

from RAG import classify_query, hybrid_search, generate_answer


DATASET_PATH = Path(__file__).with_name("insurance_dataset.json")

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    DATA = json.load(file)

PATIENTS = DATA.get("patients", [])
POLICIES = DATA.get("policies", [])


def find_patient(query):
    query_lower = query.lower()

    for patient in PATIENTS:
        name = str(patient.get("name", "")).strip().lower()
        patient_id = str(patient.get("patient_id", "")).strip().lower()

        if name and name in query_lower:
            return patient

        if patient_id and patient_id in query_lower:
            return patient

    return None


def is_hybrid_rag_question(query):
    q = query.lower()
    terms = (
        "hybrid rag",
        "bm25",
        "qdrant",
        "reciprocal rank fusion",
        "rrf",
        "sentence transformer",
        "semantic search",
        "keyword search",
    )
    return any(term in q for term in terms)


def hybrid_rag_explanation():
    return (
        "MEDI CLEAR AI uses Hybrid RAG to retrieve relevant information. "
        "BM25 finds records using keyword matching. Sentence Transformers "
        "converts text into embeddings, and Qdrant searches for semantically "
        "similar records. Reciprocal Rank Fusion (RRF) combines the rankings "
        "from BM25 and Qdrant. The highest-ranked document is then supplied "
        "to the Qwen language model to generate an answer."
    )


def find_policy(query):
    q = query.lower()

    for policy in POLICIES:
        policy_id = str(policy.get("policy_id", "")).strip().lower()
        policy_name = str(policy.get("policy_name", "")).strip().lower()

        if policy_id and re.search(
            rf"(?<!\w){re.escape(policy_id)}(?!\w)", q
        ):
            return policy

        if policy_name and policy_name in q:
            return policy

    return None


def agent_query(query):
    """Route questions with validation before generating answers."""

    query = query.strip()

    if not query:
        return "Please enter a question."

    # Answer questions about this application's architecture directly.
    if is_hybrid_rag_question(query):
        print("Agent → Project Architecture")
        return hybrid_rag_explanation()

    query_type = classify_query(query)

    if query_type == "patient":
        print("Agent → Patient Verification")

        patient = find_patient(query)

        if patient is None:
            return (
                "I could not find this patient in the available dataset. "
                "Please check the patient's name or ID."
            )

        print("Agent → Hybrid RAG")
        context = hybrid_search(query)

        if not context:
            return (
                "I could not retrieve reliable information for this patient."
            )

        return generate_answer(query, context)

    if query_type == "policy":
        print("Agent → Policy Verification")

        policy = find_policy(query)

        if policy is None:
            return (
                "I could not find a matching insurance policy in the "
                "available dataset. Please provide a valid policy name or ID."
            )

        context = hybrid_search(query)

        if not context:
            return (
                "I could not retrieve reliable information for this policy."
            )

        return generate_answer(query, context)

    print("Agent → General LLM")
    return generate_answer(query)
