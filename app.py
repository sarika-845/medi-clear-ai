import streamlit as st
from RAG import rag_query

st.set_page_config(
    page_title="MEDI CLEAR AI",
    page_icon="🏥"
)

st.title("🏥 MEDI CLEAR AI")
st.caption("Hospital Insurance & Billing Intelligence")

query = st.text_input(
    "Enter your query",
    placeholder="What is the claim status for Priya Sharma?"
)

if st.button("Generate Response"):

    if query.strip():

        with st.spinner("Generating response..."):
            answer = rag_query(query)

        st.subheader("Response")
        st.write(answer)

    else:
        st.warning("Please enter a question.")
        