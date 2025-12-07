import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "chunks", "chunks.json")
INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

def create_vector_store():
    """
    Creates a FAISS vector store from the chunks.json file using batch processing.
    """
    if not os.path.exists(CHUNKS_FILE):
        print(f"Chunks file not found at {CHUNKS_FILE}")
        return None

    # Use local embeddings
    print("Initializing HuggingFace embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Loading chunks from JSON...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_chunks = len(data)
    print(f"Loaded {total_chunks} chunks. Starting batch indexing...")
    
    # Batch size can be larger for local embeddings
    batch_size = 5000 
    vector_store = None
    
    for i in range(0, total_chunks, batch_size):
        batch_data = data[i : i + batch_size]
        batch_docs = [
            Document(page_content=chunk["content"], metadata={"source": chunk["source"], "id": chunk["id"]})
            for chunk in batch_data
        ]
        
        if vector_store is None:
            vector_store = FAISS.from_documents(batch_docs, embeddings)
        else:
            vector_store.add_documents(batch_docs)
            
        print(f"Indexed {min(i + batch_size, total_chunks)} / {total_chunks} chunks...")
        
        # Save periodically
        if (i + batch_size) % 10000 == 0 or (i + batch_size) >= total_chunks:
             if vector_store:
                print(f"Saving intermediate index to {INDEX_DIR}...")
                vector_store.save_local(INDEX_DIR)

    # Save index
    if vector_store:
        print(f"Saving index to {INDEX_DIR}...")
        vector_store.save_local(INDEX_DIR)
    
    return vector_store

def load_vector_store():
    # Use local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(INDEX_DIR):
        # print("Loading existing FAISS index...") # Reduce noise
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        return None

# The retrieve_documents function is no longer used by the new main block,
# but keeping it for potential external use or if the user wants to revert.
def retrieve_documents(query, k=5):
    vector_store = load_vector_store()
    if not vector_store:
        return []
    
    print(f"Searching for: '{query}'")
    docs = vector_store.similarity_search(query, k=k)
    return docs

# The original main function is replaced by the new test logic.
if __name__ == "__main__":
    # Force rebuild
    if os.path.exists(INDEX_DIR):
        print(f"Removing existing index at {INDEX_DIR} for a full rebuild...")
        shutil.rmtree(INDEX_DIR)
        
    create_vector_store()
    
    # Test loading
    print("\nLoading index for verification...")
    vector_store = load_vector_store()
    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        query = "What is the revenue?"
        print(f"Searching for: '{query}'")
        docs = retriever.invoke(query)
        print("\nTop 5 results:")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {doc.metadata['source']}")
            print(f"Content: {doc.page_content[:200]}...")
    else:
        print("Failed to load vector store.")


