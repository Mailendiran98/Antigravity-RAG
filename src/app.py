import streamlit as st
import os
from rag_pipeline import get_rag_chain
from retriever import load_vector_store
from dotenv import load_dotenv

# Fix OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

st.set_page_config(page_title="Financial RAG", page_icon="💰")

st.title("💰 Financial Report Q&A")
st.markdown("Ask questions about company financial statements.")

@st.cache_resource
def initialize_rag():
    vector_store = load_vector_store()
    if not vector_store:
        return None
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return get_rag_chain(retriever)

rag_chain = initialize_rag()

if not rag_chain:
    st.error("Failed to load vector store. Please run ingestion and indexing first.")
else:
    question = st.text_input("Enter your question:")
    
    if st.button("Ask"):
        if question:
            with st.spinner("Thinking..."):
                try:
                    # Invoke chain
                    response = rag_chain.invoke(question)
                    
                    st.subheader("Answer")
                    st.write(response)
                    
                    # Note: To show sources, we'd need to modify rag_pipeline to return them 
                    # in the LCEL chain (e.g. using RunnableMap).
                    # For now, we just show the answer.
                    
                except Exception as e:
                    st.error(f"Error: {e}")
