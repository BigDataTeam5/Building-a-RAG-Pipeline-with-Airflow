import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.utils import openai_token_count
import chromadb
from chromadb.utils import embedding_functions
import openai

# =========================================================================
# Make sure you have a valid "sk-..." key here:
openai.api_key = "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"
# =========================================================================

# OpenAI model settings
OPENAI_MODEL = "gpt-4"     # or "gpt-3.5-turbo"
TOP_K = 5                  # Number of relevant chunks to retrieve

# ChromaDB settings
COLLECTION_NAME = "rag_documents"
PERSIST_DIRECTORY = "./chroma_db"

# Embedding model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def fetch_url_content(url):
    """
    Fetch content from a URL.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching URL: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None


def chunk_document(text):
    """
    Split document text into chunks using RecursiveTokenChunker.
    """
    chunker = RecursiveTokenChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return chunker.split_text(text)


def embed_texts(texts):
    """
    Generate embeddings for text chunks.
    """
    return embedding_model.encode(texts)


def get_or_create_collection():
    """
    Get or create a ChromaDB collection.
    Compatible with ChromaDB v0.6.0, 
    where list_collections() returns a list of strings (collection names).
    """
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    existing_collections = client.list_collections()

    if COLLECTION_NAME in existing_collections:
        print(f"Using existing collection: {COLLECTION_NAME}")
        return client.get_collection(name=COLLECTION_NAME)
    else:
        print(f"Creating new collection: {COLLECTION_NAME}")
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


def add_chunks_to_collection(chunks, source_id):
    """
    Add document chunks to the ChromaDB collection.
    """
    collection = get_or_create_collection()
    if not chunks:
        return 0

    ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_id, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)


def retrieve_relevant_chunks(query):
    """
    Retrieve the most relevant document chunks for a query.
    """
    collection = get_or_create_collection()
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    sources = [m["source"] for m in metadatas]

    print(f"Retrieved {len(chunks)} chunks for query: '{query}'")
    for i, (chunk, source, distance) in enumerate(zip(chunks, sources, distances)):
        print(f"\nChunk {i+1} (Distance: {distance:.4f}, Source: {source}):")
        print(chunk[:300], "...")
    return chunks, sources


def generate_response(query, chunks, sources):
    """
    Generate a response using OpenAI API based on retrieved chunks.
    """
    if not chunks:
        return "No relevant information found."

    context_with_sources = [
        f"Source [{i+1}] ({src}): {chk}"
        for i, (chk, src) in enumerate(zip(chunks, sources))
    ]
    context = "\n\n".join(context_with_sources)

    # You can add a system message if you like
    system_message = (
        "You are a helpful AI assistant that provides accurate answers "
        "based on the given context. "
        "If the context does not contain relevant information, "
        "acknowledge that and avoid making up information. "
        "Cite your sources using the source numbers provided."
    )

    user_message = (
        f"Question: {query}\n\n"
        f"Context information:\n{context}\n\n"
        "Please provide a factually accurate response based on the context."
    )

    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        # v1.0.0+ of openai: 
        # Access as an attribute, not subscript
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"


def rag_pipeline(url, query):
    """
    Full RAG pipeline: fetch the remote URL, chunk, store, retrieve, and generate a response.
    """
    print(f"Step 1: Fetching document from URL:\n{url}")
    content = fetch_url_content(url)
    if not content:
        return "Failed to fetch document."

    print("Step 2: Chunking document")
    chunks = chunk_document(content)
    print(f"✅ Created {len(chunks)} chunks")

    print("Step 3: Adding chunks to vector store")
    num_added = add_chunks_to_collection(chunks, url)
    print(f"✅ Added {num_added} chunks")

    print(f"Step 4: Retrieving relevant chunks for query: '{query}'")
    relevant_chunks, sources = retrieve_relevant_chunks(query)

    print("Step 5: Generating response")
    final_answer = generate_response(query, relevant_chunks, sources)
    return final_answer


if __name__ == "__main__":
    # Example usage with your Q1 .md link
    q1_url = (
        "https://rag-pipeline-with-airflow.s3.us-east-2.amazonaws.com/docling_markdowns/2020/q1.md?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEC0aCXVzLWVhc3QtMiJHMEUCIQDrIEbp%2F5Uhc3tVL1J7C9VZ92Whr4TW4YxnDNGxgiUxdwIgB2EmzbUe%2BPvcGFPoV5J7pEeQd3wmuH3%2FVUrwblGsPZ4q1AMIhv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw5NTQ5NzYzMTYxMjQiDP3%2F8rfBEd9b31Li%2FiqoA1HlX2fTJL6MDsmxQIkJXONWuOItGLA9slRkfiNkHaz6fiGULzhBp59bv6DnMAGffAE%2BTxywS04wK01Ocq3tsqaTn3PLv65hvmcVz%2FzjTIeumTWaRoHe%2FalJoZPcnMgmL7Lus1Si9tlM5wmI1%2BRHIOam%2ByNLRpLoCADkfjcfW3L%2BVrDdQRpLQjfgGF4CWRFJNtvflXTTCA47rPTbHS9hr9WSzra2M1T1OIOvC0HFbep6PdZKbgxw0FfPr1IO9EVmsY253KXTs6TCs95DCMGCLo6jwfc626dp5035rtKeTSThHqNH2NJC7Cstr6ym41lC4Rv1kWrM6JHuNs6SOEbF3OC7Jj25AorDF4bVNjpPcGoO6g%2B7St5Y4oO105wvSc38RrSqtaQSAgDrYMXVNmvO91Ia3GMZF2gj6k%2BvES0nrkc0Gy7qfGD%2BVV5tXVsjMXwykkZrVnO4Zy59vD2YXM3vpMet7eR6J9xXliPZHDZP30AZATCoF2Zrs2VqJaxQsCdncRkiMx2VUUNIMeRvm%2F%2F5hkJjLVj%2FPGR4dbGoEim57LyKkR3B2OizOwwwiP7svgY65AKNWt4l%2F2g9AVWuuFKI9f%2B4PGD%2FYhj7wyxjYGHZ5smV4XX1iINJrdtXYrh2zuntD9bK1MUiedCbrnwtpqkEj579wuLx%2FQO7pSuK%2Fs6v6H5BwtFBX29WMx%2B4oF5IZGsyxQYEFFwBPGMR8dJoZ8szRn9J8t45ny9IFtpRRbCptOI9pYaW8OWZH4XVjs9oeQKdC%2BPaTAcBbgMM9C8ZJcS8Yj9QIICWZLCEjZIigCiNDWb8pqAJcA4YBnZC2WrlrBeax1d2RvOkIAssL9gnlTLFPxvwBzjxQIJ6hBX8G3UtG8EXXm6Z4KNKUMGEqkOwcwYpL9WRj7gAVoiR1kwyeWyWay5UnQLwtRQpZ3xg6jxvrJ79QOZSSiy8C833UZW6pYZbRsgSm%2BxhyuRz91TGQTIicvzMaGjrhdez5gyNT0KZaI4pZcH3j%2BvEvwyCxJxsozgq3ZsDd1DICFm1yG%2FjgNP%2F3YvxuX7Q9Q%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA54WIGFLOD5HOHMIY%2F20250320%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20250320T051955Z&X-Amz-Expires=7200&X-Amz-SignedHeaders=host&X-Amz-Signature=b4a10b3809e4dadc43b2cca2811ba798460329914201d6f846dcfea77ae6911e"
        # truncated for brevity
    )

    test_query = "What is Nvidia?"
    final_answer = rag_pipeline(q1_url, test_query)

    print("\nFinal Answer:")
    print("=" * 80)
    print(final_answer)
    print("=" * 80)
