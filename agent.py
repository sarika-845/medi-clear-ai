from RAG import classify_query, hybrid_search, generate_answer


def agent_query(query):

    query_type = classify_query(query)

    if query_type in ["patient", "policy"]:
        print("Agent → Hybrid RAG")

        context = hybrid_search(query)

        if not context:
            return "I could not find the requested information."

        return generate_answer(query, context)

    print("Agent → General LLM")
    return generate_answer(query)