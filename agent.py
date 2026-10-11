import json
from pathlib import Path

from RAG import classify_query, hybrid_search, generate_answer


# Load the existing patient dataset
DATASET_PATH = Path(__file__).with_name("insurance_dataset.json")

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    DATA = json.load(file)

PATIENTS = DATA.get("patients", [])


def find_patient(query):
    """Return a patient only when their name or ID appears in the query."""

    query_lower = query.lower()

    for patient in PATIENTS:
        name = str(patient.get("name", "")).strip().lower()
        patient_id = str(patient.get("patient_id", "")).strip().lower()

        if name and name in query_lower:
            return patient

        if patient_id and patient_id in query_lower:
            return patient

    return None


def agent_query(query):
    """Route queries through patient validation, Hybrid RAG, or the LLM."""

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
        print("Agent → Policy Hybrid RAG")

        context = hybrid_search(query)

        if not context:
            return (
                "I could not find the requested policy information "
                "in the available dataset."
            )

        return generate_answer(query, context)

    print("Agent → General LLM")
    return generate_answer(query)
