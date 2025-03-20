import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import openai
from chunking_evaluation.chunking import RecursiveTokenChunker, KamradtModifiedChunker
from chunking_evaluation.utils import openai_token_count

# =========================================================================
# OpenAI API Key Setup
# =========================================================================
openai.api_key = os.getenv("OPENAI_API_KEY", "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")  # Replace if needed

# OpenAI model settings
OPENAI_MODEL = "gpt-4"
TOP_K = 5  # Number of relevant chunks to retrieve

# ChromaDB settings
COLLECTION_NAME = "rag_documents"
PERSIST_DIRECTORY = "./chroma_db"

# Embedding model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Initialize SentenceTransformer for embedding generation
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def fetch_url_content(url):
    """Fetch content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching URL: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None


def chunk_document_simple(text):
    """Simple fixed-size chunking with overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP  # Move pointer with overlap

    return chunks


def chunk_document_recursive(text):
    """Recursive chunking using RecursiveTokenChunker."""
    chunker = RecursiveTokenChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]  # Delimiters
    )
    return chunker.split_text(text)


def chunk_document_semantic(text):
    """Semantic chunking using KamradtModifiedChunker with OpenAI embeddings."""
    api_key = openai.api_key
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable is not set. Skipping semantic chunking.")
        return chunk_document_simple(text)  # Fallback

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, 
        model_name="text-embedding-3-small"
    )

    kamradt_chunker = KamradtModifiedChunker(
        avg_chunk_size=300,  # Target size in tokens
        min_chunk_size=50,   # Minimum chunk size
        embedding_function=embedding_function
    )

    return kamradt_chunker.split_text(text)


def get_or_create_collection():
    """Get or create a ChromaDB collection."""
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    existing_collections = client.list_collections()

    if COLLECTION_NAME in existing_collections:
        return client.get_collection(name=COLLECTION_NAME)
    else:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


def add_chunks_to_collection(chunks, source_id):
    """Add document chunks to ChromaDB."""
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
    """Retrieve the most relevant document chunks for a query."""
    collection = get_or_create_collection()
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    sources = [m["source"] for m in metadatas]

    return chunks, sources


def generate_response(query, chunks, sources):
    """Generate a response using OpenAI API based on retrieved chunks."""
    if not chunks:
        return "No relevant information found."

    context_with_sources = [
        f"Source [{i+1}] ({src}): {chk}"
        for i, (chk, src) in enumerate(zip(chunks, sources))
    ]
    context = "\n\n".join(context_with_sources)

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
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {str(e)}"


def rag_pipeline(url, query, chunking_strategy):
    """Full RAG pipeline: fetch, chunk, store, retrieve, and generate a response."""
    print(f"Fetching document from URL:\n{url}")
    content = fetch_url_content(url)
    if not content:
        return "Failed to fetch document."

    print(f"Chunking document using {chunking_strategy} strategy...")
    
    if chunking_strategy == "simple":
        chunks = chunk_document_simple(content)
    elif chunking_strategy == "recursive":
        chunks = chunk_document_recursive(content)
    elif chunking_strategy == "semantic":
        chunks = chunk_document_semantic(content)
    else:
        return "Invalid chunking strategy selected."

    print(f"✅ Created {len(chunks)} chunks")

    print("Adding chunks to vector store")
    num_added = add_chunks_to_collection(chunks, url)
    print(f"✅ Added {num_added} chunks")

    print(f"Retrieving relevant chunks for query: '{query}'")
    relevant_chunks, sources = retrieve_relevant_chunks(query)

    print("Generating response...")
    return generate_response(query, relevant_chunks, sources)


if __name__ == "__main__":
    # Example document URL
    q1_url = "https://rag-pipeline-with-airflow.s3.us-east-2.amazonaws.com/mistral_markdowns/2020/Q1.md?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDwaCXVzLWVhc3QtMiJHMEUCIF1kdmb%2Fwkrf8NMhmhmjJmbFaKQj5XskCeQG%2BALUfLJFAiEA88Kr37mZm4q1ed7EcCQaG4XHQh68PfUMSg7jCy3mOgMq1AMIlv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw5NTQ5NzYzMTYxMjQiDD%2FmEEUOeOqNdJ%2FY5CqoA26q6iXfHDvoDNQ77mzN%2BUbpqoP6querv67gyu2mZ1IV4CWE9i6ciVnkOYXdboG2dAeL7ktChJ5tHZxy3Bqh5SCmFweYO3YRZRtMt5tQzTvLoZUz5UyEvsAmKQ%2FsBn6mI5K%2FQ42qXpoDXj71lr30xpVBaoyUvDc04xh74l6zM%2FonJJ0mjVZ1EycaUFBTrdkcRiqj7Uw1rb%2F8DT%2FiL7bwLYMulSffgxFprHeof7sQhSiiW3LHFX3lo3vXzvok%2BXICtouOaC3YPc8bXlIRuwesahzRjlU0ZDjv2biDYI%2F6UVlfujDFEy09hu0QLahgINa6%2B51Y%2BfHkKFTmWrkhHpG79%2B72OQaUyPAEa%2Bk9J2SnwzXDOu2Z6WOD0GavULL72ul8p867PVE6YKGrss2yiJs7HHychcV4qUoNR7PaA1fOrTIYeAy9vwc2HS5rbeQGDprazBfah%2FMn3xVjiWMeNoN%2FLMiMz6VnEXvnChd%2BwOvT45j8R35nWLyr%2FcsCsz9ut763vzyu973l6pOupOuurSa2YSI0z4Yw3tULT2QXJ8DQkUS0KP1F23%2FHcD4wv6DxvgY65AIiBBnj1gz7zfyF8WI6WMGsFyR%2F1TpIy%2BbtI4edN9PeP1NrPT4tKNRcLYKfquD%2BtiaPPpTbcIBWHKcM1PpRCs7QeFSioRTM5idg%2BAy6u3rI%2FUeDi%2BaE3qeqskGsUkL5PHtffumHi2JmZJk2ML3tFLgRU32pFUIriNjBx7YywoEHVplj1JdKaOjjYQCghElvV7c4%2FdEOgo12ol5cVbzI8O7%2FoZik5lRU%2BUqBX9WH2TYOlDPLuni%2FGM79xmlNXARxYdLkcikYASHsAnVswRM6cPJHbZAqPJqY1Fka4u3kIOKrfIHkRDE%2FIU%2BjUuwJKgOPDRwmln3vKqxVGhJHSNferKa0JO5wWrtllrtdhdCEldyp4VD%2F9o03l65hMZ5wJn%2BvXP7bnB5Md3ghBLw2nxin%2FkO%2BHaI4SrJJCm0hm4RQvdbXy1e3%2BqLlZ5n6%2FBOu8ye0qzDxhO%2FBhsrIlIMdRcWmT%2F4%2F8aUu1g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA54WIGFLOPXXYXS5D%2F20250320%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20250320T202523Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=5859999e6baa41c1dff99a00c1e31fb7a0934bf492ff1c1df5fd2380f45a6dba"  # Replace with an actual document URL
    test_query = "What is Nvidia?"

    # Ask user for preferred chunking strategy
    print("\nChoose a chunking strategy:")
    print("1. Simple Chunking")
    print("2. Recursive Chunking")
    print("3. Semantic Chunking (Kamradt's Method)")

    chunking_choice = input("Enter your choice (1, 2, or 3): ").strip()

    if chunking_choice == "1":
        chunking_strategy = "simple"
    elif chunking_choice == "2":
        chunking_strategy = "recursive"
    elif chunking_choice == "3":
        chunking_strategy = "semantic"
    else:
        print("Invalid choice. Defaulting to Recursive Chunking.")
        chunking_strategy = "recursive"

    print(f"\n=== RAG Pipeline ({chunking_strategy.capitalize()} Chunking) ===")
    final_answer = rag_pipeline(q1_url, test_query, chunking_strategy)

    print("\nFinal Answer:")
    print("=" * 80)
    print(final_answer)
    print("=" * 80)
