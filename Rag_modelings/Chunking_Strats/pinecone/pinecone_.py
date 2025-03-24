import json
import openai
from uuid import uuid4  # For unique IDs
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import numpy as np
import tiktoken

# Load environment variables
load_dotenv()

# Initialize API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
INDEX_NAME = "majestic-ash"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENV
        )
    )
index = pc.Index(INDEX_NAME)

# Function to generate embeddings
def get_embedding(text: str) -> list:
    """Generate embedding using the new OpenAI API format"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

# Load chunks from JSON file
def load_chunks_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data.get("chunks", [])  # Access the "chunks" key

def truncate_text(text, max_bytes=1000):
    """Truncate text to ensure it doesn't exceed max_bytes"""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore')

def batch_vectors(vectors, batch_size=100):
    """Yield vectors in batches"""
    for i in range(0, len(vectors), batch_size):
        yield vectors[i:i + batch_size]

def upload_to_pinecone(chunks):
    vectors = []
    for chunk in chunks:
        text = chunk.get("text", "")
        filename = chunk.get("file_path", "")
        chunk_id = str(chunk.get("id", str(uuid4())))
        
        try:
            vector = get_embedding(text)
            # Truncate text for metadata to avoid size limits
            truncated_text = truncate_text(text)
            vectors.append({
                "id": chunk_id,
                "values": vector,
                "metadata": {
                    "text_preview": truncated_text,  # Store truncated version
                    "file_name": filename,
                    "original_length": len(text)  # Store original length for reference
                }
            })
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {str(e)}")
            continue

    # Upload vectors in batches
    total_uploaded = 0
    for batch in batch_vectors(vectors):
        try:
            index.upsert(batch)
            total_uploaded += len(batch)
            print(f"Uploaded batch of {len(batch)} vectors. Total uploaded: {total_uploaded}")
        except Exception as e:
            print(f"Error uploading batch: {str(e)}")
            continue

    print(f"Finished uploading. Total vectors uploaded: {total_uploaded}")

def get_query_embedding(query_text: str) -> list:
    """Generate embedding for the query text"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query_text]
    )
    return response.data[0].embedding

def search_pinecone(query: str, top_k: int = 3):
    """Search Pinecone index for relevant chunks"""
    query_embedding = get_query_embedding(query)
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def generate_response(query: str, context_chunks: list) -> dict:
    """Generate LLM response using retrieved chunks as context"""
    # Combine chunks into context
    context = "\n".join([chunk['metadata']['text_preview'] for chunk in context_chunks])
    
    # Create prompt with query and context
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    # Count input tokens
    input_tokens = sum(count_tokens(msg["content"]) for msg in messages)
    
    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    output_tokens = count_tokens(response.choices[0].message.content)
    
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    }

def interactive_qa():
    """Interactive Q&A loop"""
    print("\nWelcome to the Document Q&A System!")
    print("Enter your questions (or 'quit' to exit)")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        try:
            # Search for relevant chunks
            results = search_pinecone(query)
            
            if not results['matches']:
                print("No relevant information found.")
                continue
            
            # Generate response using chunks
            response_data = generate_response(query, results['matches'])
            
            print("\nAnswer:", response_data["content"])
            print("\nToken Usage:")
            print(f"Input tokens: {response_data['usage']['input_tokens']}")
            print(f"Output tokens: {response_data['usage']['output_tokens']}")
            print(f"Total tokens: {response_data['usage']['total_tokens']}")
            
            # Optionally show sources
            print("\nSources:")
            for i, match in enumerate(results['matches'], 1):
                print(f"{i}. Score: {match['score']:.4f}")
                print(f"   File: {match['metadata'].get('file_name', 'Unknown')}")
                print(f"   Preview: {match['metadata']['text_preview'][:100]}...")
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")

# Add this function to handle multiple files
def process_multiple_files(file_paths: list[str]) -> None:
    """Process multiple JSON files and upload their chunks to Pinecone"""
    total_chunks = 0
    for file_path in file_paths:
        try:
            print(f"\nProcessing file: {file_path}")
            chunks = load_chunks_from_json(file_path)
            upload_to_pinecone(chunks)
            total_chunks += len(chunks)
            print(f"Successfully processed {len(chunks)} chunks from {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"\nTotal chunks processed across all files: {total_chunks}")

def view_first_10_vectors():
    """View the first 10 vectors in the Pinecone index"""
    try:
        # Query to get first 10 vectors
        results = index.query(
            vector=[0] * 1536,  # Dummy vector
            top_k=10,
            include_metadata=True
        )
        
        # Get the first 10 items
        first_10 = results.matches
        
        print("\nFirst 10 vectors in the index:")
        for i, match in enumerate(first_10, 1):
            print(f"\n{i}. Vector ID: {match.id}")
            print(f"   File: {match.metadata.get('file_name', 'Unknown')}")
            print(f"   Preview: {match.metadata['text_preview'][:100]}...")
            
    except Exception as e:
        print(f"Error fetching vectors: {str(e)}")

# Update the main section of your code
if __name__ == "__main__":
    # # List of JSON files to process
    # json_file_paths = [
    #     r"..\output\token_chunking\chunks.json",
    #     r"..\output\character_chunking\chunks.json",
    #     r"..\output\recursive_chunking\chunks.json",
    #     # Add more file paths as needed
    # ]

    # # Process all files first
    # process_multiple_files(json_file_paths)

    # Then start the interactive Q&A
    # interactive_qa()
    view_first_10_vectors()