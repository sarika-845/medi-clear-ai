import streamlit as st
from agent import agent_query

st.set_page_config(
    page_title="MEDI CLEAR AI",
    page_icon="🏥"
)

st.title("🏥 MEDI CLEAR AI")
st.caption("Agentic Hybrid RAG Insurance & Billing Assistant")

query = st.text_input(
    "Enter your query",
    placeholder="What is the claim status for Priya Sharma?"
)

if st.button("Generate Response"):

    if query.strip():

        with st.spinner("Generating response..."):
            answer = agent_query(query)

        st.subheader("Response")
        st.write(answer)

    else:
        st.warning("Please enter a question.")