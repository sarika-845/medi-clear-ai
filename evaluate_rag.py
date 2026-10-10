
import os
import json
import asyncio

from openai import AsyncOpenAI
from google import genai

from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
)

from RAG import hybrid_search, generate_answer


# --------------------------------------------------
# GEMINI SETUP
# --------------------------------------------------

API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Configure it in your terminal."
    )

OUTPUT_FILE = "ragas_results.json"
DATASET_FILE = "insurance_dataset.json"

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60,
    max_retries=0,
)

llm = llm_factory(
    "gemini-3.8-flash",
    provider="openai",
    client=client,
    adapter="instructor",
)

embedding_client = genai.Client(api_key=API_KEY)

embeddings = GoogleEmbeddings(
    client=embedding_client,
    model="gemini-embedding-001",
)


# --------------------------------------------------
# RAGAS METRICS
# --------------------------------------------------

metrics = {
    "context_precision": ContextPrecision(llm=llm),
    "context_recall": ContextRecall(llm=llm),
    "faithfulness": Faithfulness(llm=llm),
    "answer_correctness": AnswerCorrectness(
        llm=llm,
        embeddings=embeddings,
    ),
}


# --------------------------------------------------
# SAVE AND ERROR HANDLING
# --------------------------------------------------

def save_results(results):
    """Save progress safely without overwriting with partial file data."""
    temp_file = OUTPUT_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    os.replace(temp_file, OUTPUT_FILE)


def is_temporary_api_error(error):
    """Identify quota, rate-limit, and temporary server errors."""
    message = str(error).lower()

    return any(
        term in message
        for term in (
            "429",
            "503",
            "unavailable",
            "high demand",
            "quota",
            "resource_exhausted",
            "rate limit",
            "500",
            "502",
            "504",
            "internal server error",
        )
    )


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

async def main():
    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    patients = data["patients"]

    # Load existing results so completed metrics are preserved.
    results = []

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            results = json.load(file)

    saved = {
        row["question"]: row
        for row in results
    }

    stopped_early = False

    try:
        for index, patient in enumerate(patients, start=1):
            question = (
                f"What is the claim status for {patient['name']}?"
            )

            reference = (
                f"The claim status is "
                f"{patient['claim']['claim_status']}."
            )

            print(
                f"\nPatient {index}/{len(patients)}: {question}",
                flush=True,
            )

            # Reuse existing row if it was previously saved.
            row = saved.get(question)

            if row is None:
                row = {
                    "question": question,
                    "answer": "",
                    "context_precision": None,
                    "context_recall": None,
                    "faithfulness": None,
                    "answer_correctness": None,
                }

                results.append(row)
                saved[question] = row

            # Skip patients whose four metrics are complete.
            pending = [
                name
                for name in metrics
                if row.get(name) is None
            ]

            if not pending:
                print("All metrics completed. Skipping patient.")
                continue

            # Retrieve context for any unfinished metrics.
            try:
                context = hybrid_search(question)

                if not context:
                    print("No context found. Skipping patient.")
                    save_results(results)
                    continue

                # Generate the answer only if it is not already saved.
                if not row.get("answer"):
                    row["answer"] = generate_answer(
                        question,
                        context,
                    )
                    save_results(results)

            except Exception as error:
                print(f"Retrieval or answer generation failed: {error}")
                save_results(results)

                if is_temporary_api_error(error):
                    stopped_early = True
                    print("Temporary API problem. Stopping safely.")
                    break

                raise

            print(f"Answer: {row['answer']}")

            # Evaluate only missing metrics.
            for name in pending:
                print(f"Evaluating {name}...", flush=True)

                try:
                    if name in (
                        "context_precision",
                        "context_recall",
                    ):
                        score = await metrics[name].ascore(
                            user_input=question,
                            retrieved_contexts=[context],
                            reference=reference,
                        )

                    elif name == "faithfulness":
                        score = await metrics[name].ascore(
                            user_input=question,
                            response=row["answer"],
                            retrieved_contexts=[context],
                        )

                    else:
                        score = await metrics[name].ascore(
                            user_input=question,
                            response=row["answer"],
                            reference=reference,
                        )

                    row[name] = float(score.value)

                    # Persist each successful metric immediately.
                    save_results(results)

                    print(
                        f"{name}: {score.value:.3f}",
                        flush=True,
                    )

                except Exception as error:
                    # Preserve all successful scores.
                    save_results(results)

                    print(
                        f"Evaluation failed for {name}: {error}",
                        flush=True,
                    )

                    if is_temporary_api_error(error):
                        stopped_early = True
                        print(
                            "Gemini quota or temporary service limit "
                            "reached. Saved progress is preserved.",
                            flush=True,
                        )
                        break

                    raise

            if stopped_early:
                break

    finally:
        save_results(results)
        await client.close()

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("MEDI CLEAR AI - RAGAS EVALUATION SUMMARY")
    print("=" * 50)

    for name in metrics:
        scores = [
            row[name]
            for row in results
            if row.get(name) is not None
        ]

        if scores:
            average = sum(scores) / len(scores)
            print(
                f"{name}: {average:.3f} "
                f"({len(scores)} completed scores)"
            )
        else:
            print(f"{name}: No completed scores")

    completed_patients = sum(
        all(row.get(name) is not None for name in metrics)
        for row in results
    )

    print(f"\nPatients with all four metrics: {completed_patients}")
    print(f"Patients with saved results: {len(results)}")
    print(f"Results saved in: {OUTPUT_FILE}")

    if stopped_early:
        print(
            "Evaluation paused due to a temporary API problem. "
            "Run again later to resume unfinished metrics."
        )
    elif completed_patients == len(patients):
        print("All patient evaluations completed.")
    else:
        print(
            "Evaluation finished with some metrics still pending. "
            "Run again later to resume."
        )


if __name__ == "__main__":
    asyncio.run(main())
