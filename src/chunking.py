import os
import glob
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DATA_DIR = "data/processed"
CHUNKS_DIR = "data/chunks"

def load_processed_files(source_dir):
    """
    Loads text files from the processed directory.
    """
    texts = []
    files = glob.glob(os.path.join(source_dir, "*.txt"))
    # Process all files
    for file_path in files:
        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            texts.append({"source": file_path, "content": content})
    return texts

def chunk_text(text_data):
    """
    Splits text into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = []
    for item in text_data:
        source = item["source"]
        content = item["content"]
        
        split_texts = text_splitter.split_text(content)
        for i, text in enumerate(split_texts):
            chunks.append({
                "id": f"{os.path.basename(source)}_{i}",
                "source": source,
                "content": text
            })
    return chunks

def save_chunks(chunks, output_dir):
    """
    Saves chunks to a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)
    print(f"Saved {len(chunks)} chunks to {output_path}")

def test_embedding_generation(chunks):
    """
    Tests embedding generation for a single chunk.
    """
    if not chunks:
        print("No chunks to test.")
        return

    print("Testing embedding generation...")
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-ada-002"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        )
        # Test with the first chunk
        sample_text = chunks[0]["content"]
        vector = embeddings.embed_query(sample_text)
        print(f"Embedding generated successfully! Vector length: {len(vector)}")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        print("Make sure AZURE_OPENAI_ENDPOINT and OPENAI_API_KEY are set.")

def main():
    print("Loading processed files...")
    text_data = load_processed_files(PROCESSED_DATA_DIR)
    
    print(f"Chunking {len(text_data)} documents...")
    chunks = chunk_text(text_data)
    
    save_chunks(chunks, CHUNKS_DIR)
    
    test_embedding_generation(chunks)

if __name__ == "__main__":
    main()
