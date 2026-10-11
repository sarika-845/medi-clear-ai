import os
import json
import asyncio

from openai import (
    AsyncOpenAI,
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
)
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
# CONFIGURATION
# --------------------------------------------------

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Set it in your terminal."
    )

DATASET_FILE = "insurance_dataset.json"
OUTPUT_FILE = "ragas_results.json"

# Start with five patients. Increase to 24 after this run.
MAX_PATIENTS = 5

METRIC_NAMES = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_correctness",
]

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=90.0,
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
# SAVE AND LOAD RESULTS
# --------------------------------------------------

def save_results(results):
    temporary_file = OUTPUT_FILE + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    os.replace(temporary_file, OUTPUT_FILE)


def load_results():
    if not os.path.exists(OUTPUT_FILE):
        return []

    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list):
        raise ValueError(
            f"{OUTPUT_FILE} must contain a JSON list."
        )

    return results


# --------------------------------------------------
# ERROR HANDLING
# --------------------------------------------------

def is_temporary_api_error(error):
    message = str(error).lower()

    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return True

    if isinstance(error, APIStatusError):
        if error.status_code in (408, 429, 500, 502, 503, 504):
            return True

    return any(
        text in message
        for text in (
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "readtimeout",
            "unavailable",
            "high demand",
            "quota",
            "resource_exhausted",
            "rate limit",
        )
    )


async def evaluate_with_retry(metric_name, **kwargs):
    for attempt in range(3):
        try:
            return await asyncio.wait_for(
                metrics[metric_name].ascore(**kwargs),
                timeout=100,
            )

        except Exception as error:
            temporary = (
                is_temporary_api_error(error)
                or isinstance(error, asyncio.TimeoutError)
            )

            if not temporary or attempt == 2:
                raise

            delay = 10 * (2 ** attempt)

            print(
                f"Temporary failure in {metric_name}: {error}",
                flush=True,
            )
            print(
                f"Retrying in {delay} seconds "
                f"(attempt {attempt + 2}/3)...",
                flush=True,
            )

            await asyncio.sleep(delay)


# --------------------------------------------------
# EVALUATE A SINGLE METRIC
# --------------------------------------------------

async def evaluate_metric(
    name,
    question,
    answer,
    context,
    reference,
):
    if name in ("context_precision", "context_recall"):
        return await evaluate_with_retry(
            name,
            user_input=question,
            retrieved_contexts=[context],
            reference=reference,
        )

    if name == "faithfulness":
        return await evaluate_with_retry(
            name,
            user_input=question,
            response=answer,
            retrieved_contexts=[context],
        )

    return await evaluate_with_retry(
        name,
        user_input=question,
        response=answer,
        reference=reference,
    )


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

async def main():
    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    patients = dataset["patients"][:MAX_PATIENTS]

    results = load_results()
    saved = {row["question"]: row for row in results}

    paused = False

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

            for name in METRIC_NAMES:
                row.setdefault(name, None)

            pending = [
                name
                for name in METRIC_NAMES
                if row[name] is None
            ]

            if not pending:
                print("All four metrics already complete. Skipping.")
                continue

            try:
                context = hybrid_search(question)

                if not context:
                    print("No context retrieved; skipping this patient.")
                    save_results(results)
                    continue

                if not row.get("answer"):
                    row["answer"] = generate_answer(
                        question,
                        context,
                    )
                    save_results(results)

            except Exception as error:
                save_results(results)
                print(f"Retrieval or generation failed: {error}")

                if is_temporary_api_error(error):
                    paused = True
                    break

                raise

            print(f"Answer: {row['answer']}")

            for name in pending:
                print(f"Evaluating {name}...", flush=True)

                try:
                    score = await evaluate_metric(
                        name,
                        question,
                        row["answer"],
                        context,
                        reference,
                    )

                    row[name] = float(score.value)
                    save_results(results)

                    print(
                        f"{name}: {score.value:.3f}",
                        flush=True,
                    )

                except Exception as error:
                    save_results(results)

                    print(
                        f"Could not complete {name}: {error}",
                        flush=True,
                    )

                    if (
                        is_temporary_api_error(error)
                        or isinstance(error, asyncio.TimeoutError)
                    ):
                        print(
                            "Temporary API failure. "
                            "Completed scores have been saved.",
                            flush=True,
                        )
                        paused = True
                        break

                    raise

            if paused:
                break

    finally:
        save_results(results)
        await client.close()
        embedding_client.close()

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("MEDI CLEAR AI - RAGAS EVALUATION SUMMARY")
    print("=" * 50)

    for name in METRIC_NAMES:
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

    selected_questions = {
        f"What is the claim status for {p['name']}?"
        for p in patients
    }

    selected_rows = [
        row
        for row in results
        if row["question"] in selected_questions
    ]

    fully_evaluated = sum(
        all(row.get(name) is not None for name in METRIC_NAMES)
        for row in selected_rows
    )

    print(f"\nSelected patients: {len(patients)}")
    print(f"Patients with saved results: {len(selected_rows)}")
    print(f"Fully evaluated patients: {fully_evaluated}")
    print(f"Results saved in: {OUTPUT_FILE}")

    if paused:
        print("Evaluation paused. Run again to resume.")
    elif fully_evaluated == len(patients):
        print("All selected patient evaluations completed.")
    else:
        print("Some metrics remain pending. Run again to resume.")


if __name__ == "__main__":
    asyncio.run(main())
