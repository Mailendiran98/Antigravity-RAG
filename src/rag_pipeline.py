import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from retriever import load_vector_store

# Fix OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
        api_version="2024-02-15-preview",
        temperature=0,
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(retriever):
    llm = get_llm()
    
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

    Context: {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    print("Loading vector store...")
    vector_store = load_vector_store()
    if not vector_store:
        print("Failed to load vector store.")
        return
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    print("Initializing RAG chain...")
    rag_chain = get_rag_chain(retriever)
    
    query = "What is the revenue?"
    print(f"\nQuestion: {query}")
    
    try:
        response = rag_chain.invoke(query)
        print(f"\nAnswer: {response}")
        
        # Note: LCEL default chain doesn't return source docs easily without modification.
        # For verification, we can just print the answer. 
        # If sources are needed, we'd need a more complex RunnableMap.
        # For now, let's just verify the answer generation.
    except Exception as e:
        print(f"Error executing RAG chain: {e}")

if __name__ == "__main__":
    main()
