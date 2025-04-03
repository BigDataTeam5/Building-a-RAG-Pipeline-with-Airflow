import json
from uuid import uuid4  
from dotenv import load_dotenv
import os
import sys
from anthropic import Anthropic
import numpy as np
import hashlib
from pathlib import Path
import nltk
import time
from nltk.corpus import stopwords
import sqlite3
import tiktoken
import sqlite3
import pickle
import logging
from chunking_evaluation.chunking import (
    FixedTokenChunker,
    RecursiveTokenChunker,
    KamradtModifiedChunker,
    ClusterSemanticChunker,
    LLMSemanticChunker
)

# Remove the global tiktoken initialization
encoding = tiktoken.get_encoding("cl100k_base")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
# load english stopwords from nltk
STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update(['would', 'could', 'should', 'might', 'many', 'much'])

def initialize_anthropic_client():
    """Initialize the Anthropic client."""
    load_dotenv(override=True)
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("Missing Anthropic API key in environment variables")
    return Anthropic(api_key=anthropic_api_key)

def initialize_local_db(db_path="local_vectors.db"):
    """Initialize SQLite database for storing vectors."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            id TEXT PRIMARY KEY,
            vector BLOB,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"Initialized local database at {db_path}")


def save_vectors_to_local_db(vectors, db_path="local_vectors.db"):
    """Save vectors to SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for vector in vectors:
        vector_id = vector["id"]
        vector_blob = pickle.dumps(vector["values"])  # Serialize vector
        metadata = json.dumps(vector["metadata"])  # Serialize metadata
        cursor.execute("""
            INSERT OR REPLACE INTO vectors (id, vector, metadata)
            VALUES (?, ?, ?)
        """, (vector_id, vector_blob, metadata))
    conn.commit()
    conn.close()
    print(f"Saved {len(vectors)} vectors to local database.")

import numpy as np

# Replace the tiktoken initialization with a function that loads it on demand
def get_tiktoken_encoding():
    """Get tiktoken encoding with error handling and memory management."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except (ImportError, MemoryError) as e:
        logger.warning(f"Failed to load tiktoken: {str(e)}. Using fallback character counting.")
        return None

# Update the character_based_chunking function
def character_based_chunking(text, chunk_size=400, overlap=50):
    """Character-based chunking with fallback if tiktoken fails."""
    try:
        # Get encoding safely
        enc = get_tiktoken_encoding()
        
        if (enc is None):
            # Fallback to character-based chunking without tiktoken
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
            logger.info(f"Created {len(chunks)} chunks using fallback character-based chunking")
            return chunks
            
        # Use FixedTokenChunker with safe token counting
        token_size = chunk_size // 4
        chunker = FixedTokenChunker(
            chunk_size=token_size,
            chunk_overlap=overlap // 4,
            length_function=lambda t: len(enc.encode(t)) if enc else len(t)
        )
        
        chunks = chunker.split_text(text)
        logger.info(f"Created {len(chunks)} chunks using token-based chunking")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in character-based chunking: {str(e)}")
        # Ultimate fallback
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

# Modify the recursive_chunking function to preserve base64 images
def recursive_chunking(text, chunk_size=400, overlap=50):
    """Recursive chunking using RecursiveTokenChunker with special handling for base64 images."""
    try:
        # First, detect and extract base64 images
        import re
        base64_pattern = r'(!\[.*?\]\(data:image\/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+\))'
        
        # Split the text by images
        parts = re.split(base64_pattern, text)
        
        # Process each part
        chunks = []
        for i, part in enumerate(parts):
            # Check if this part is a base64 image
            if i % 2 == 1 and part.startswith('![') and 'base64' in part:
                # Images are kept as separate chunks with no splitting
                chunks.append(part)
                logger.info("Preserved base64 image as a separate chunk")
            else:
                # For text content, use standard recursive chunking
                token_size = chunk_size // 4
                
                chunker = RecursiveTokenChunker(
                    chunk_size=token_size,
                    chunk_overlap=overlap // 4,
                    length_function=lambda text: len(encoding.encode(text)),
                    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
                )
                
                text_chunks = chunker.split_text(part)
                chunks.extend(text_chunks)
        
        # Add a maximum chunk limit
        max_chunks = 1000
        if len(chunks) > max_chunks:
            logger.warning(f"Too many chunks generated ({len(chunks)}). Limiting to {max_chunks} chunks.")
            chunks = chunks[:max_chunks]
            
        logger.info(f"Created {len(chunks)} chunks using recursive chunking with image preservation")
        return chunks
    except Exception as e:
        logger.error(f"Error in recursive chunking: {str(e)}")
        # Fall back to simple chunking without special image handling
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]                
        
def semantic_chunking(text, avg_size=300, min_size=50):
    """Simplified semantic chunking with better fallback"""
    try:
        # Convert sizes from characters to approximate tokens
        token_avg_size = avg_size // 4  # Rough estimation of chars to tokens
        token_min_size = min_size // 4  # Rough estimation of chars to tokens
        
        # Use KamradtModifiedChunker with simpler parameters
        chunker = KamradtModifiedChunker(
            desired_chunk_size=token_avg_size,
            min_chunk_size=token_min_size,
            token_encoding=encoding
        )
        
        chunks = chunker.chunk_text(text)
        
        # Add a maximum chunk limit to prevent memory issues
        max_chunks = 500
        if len(chunks) > max_chunks:
            logger.warning(f"Too many chunks generated ({len(chunks)}). Limiting to {max_chunks}")
            chunks = chunks[:max_chunks]
            
        logger.info(f"Created {len(chunks)} chunks using semantic chunking")
        return chunks
    except Exception as e:
        logger.error(f"Error in Kamradt semantic chunking: {str(e)}")
        # Fallback to recursive chunking
        return recursive_chunking(text, avg_size, min_size)

def chunk_document(text, chunking_strategy):
    """Chunk document using the specified strategy."""
    # Normalize chunking strategy name
    strategy_lower = chunking_strategy.lower().replace("-", "").replace(" ", "_")
    
    if "character" in strategy_lower or strategy_lower == "characterbased_chunking":
        return character_based_chunking(text)
    elif "recursive" in strategy_lower or strategy_lower == "recursive_chunking":
        return recursive_chunking(text)
    elif "semantic" in strategy_lower or strategy_lower == "semantic_chunking" or "kamradt" in strategy_lower:
        return semantic_chunking(text)
    else:
        try:
            # Use ClusterSemanticChunker if explicitly requested
            chunker = ClusterSemanticChunker(token_encoding=encoding)
            chunks = chunker.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks using cluster semantic chunking")
            return chunks
        except Exception as e:
            logger.error(f"Error in cluster semantic chunking: {str(e)}")
            return semantic_chunking(text)
    
        
# Import LiteLLM response generator
sys.path.append("Backend")
from litellm_query_generator import generate_response, MODEL_CONFIGS
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("rag_pipeline")

# Task 0: Implement conditional chunking
def process_document_with_chunking(text, chunking_strategy):
    """Process document using the specified chunking strategy."""
    try:
        logger.info(f"Chunking document using {chunking_strategy} strategy...")
        
        # Add debug info
        logger.info(f"Input text length: {len(text)} characters")
        
        # Safety check for empty text
        if not text or text.isspace():
            logger.warning("Input text is empty or whitespace only")
            return ["Empty document"]
            
        chunks = chunk_document(text, chunking_strategy)
        return chunks
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        raise

        
        
# Add function to extract base64 images from markdown text
def extract_base64_images(text):
    """Extract base64 images from markdown content"""
    import re
    
    # Pattern to match markdown image syntax with base64 content
    pattern = r'!\[.*?\]\(data:image\/[a-zA-Z]+;base64,([a-zA-Z0-9+/=]+)\)'
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    # Store extracted images
    images = []
    for i, base64_str in enumerate(matches):
        try:
            # Store image data
            images.append({
                "id": f"img_{i}",
                "base64_data": base64_str,
                "format": "base64"
            })
            logger.info(f"Extracted base64 image #{i+1}")
        except Exception as e:
            logger.error(f"Error processing base64 image: {str(e)}")
    
    return images

# Update save_chunks_to_json to handle base64 images
def save_chunks_to_json(chunks, index_name):
    """Save full chunks to a JSON file for retrieval during querying
    
    Args:
        chunks (list): List of document chunks with text content
        index_name (str): Name of the Pinecone index for reference
    
    Returns:
        str: Path to the saved JSON file
    """
    try:
        # Create chunks directory if it doesn't exist
        chunks_dir = Path("chunk_storage")
        chunks_dir.mkdir(exist_ok=True)
        
        # Create a filename based on the index name
        json_path = chunks_dir / f"{index_name}_chunks.json"
        
        # Prepare chunks data with unique IDs
        chunks_data = {}
        for i, chunk in enumerate(chunks):
            # Generate a unique ID for the chunk
            if isinstance(chunk, str):
                text = chunk
                # Create a hash-based ID for consistent retrieval
                chunk_id = hashlib.md5(text.encode('utf-8')).hexdigest()
                
                # Extract base64 images if any
                images = extract_base64_images(text)
                
                # Store the full chunk with its ID
                chunk_data = {
                    "text": text,
                    "index": i,
                    "length": len(text)
                }
                
                # Add images if found
                if images:
                    chunk_data["images"] = images
                    
                chunks_data[chunk_id] = chunk_data
            else:
                # Handle dictionary chunk format
                text = chunk.get("text", "")
                chunk_id = str(chunk.get("id", hashlib.md5(text.encode('utf-8')).hexdigest()))
                
                # Extract base64 images if any
                images = extract_base64_images(text)
                
                # Create base chunk data
                chunk_data = {
                    "text": text,
                    "index": i,
                    "length": len(text)
                }
                
                # Add images if found
                if images:
                    chunk_data["images"] = images
                
                # Store with any additional fields from the original chunk
                for key, value in chunk.items():
                    if key not in ["text", "id"]:
                        chunk_data[key] = value
                        
                chunks_data[chunk_id] = chunk_data
        
        # Save to JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks_data)} full chunks to {json_path}")
        return str(json_path)
    
    except Exception as e:
        logger.error(f"Error saving chunks to JSON: {str(e)}")
        return None
        
# Add a function to load chunks from the JSON file
def load_chunks_from_json(chunk_id, index_name):
    """Load a specific chunk from the JSON file by ID
    
    Args:
        chunk_id (str): ID of the chunk to retrieve
        index_name (str): Name of the Pinecone index for reference
    
    Returns:
        str: Full text content of the chunk
    """
    try:
        # Construct the path to the chunks JSON file
        chunks_dir = Path("chunk_storage")
        json_path = chunks_dir / f"{index_name}_chunks.json"
        
        # Check if the file exists
        if not json_path.exists():
            logger.warning(f"Chunks file not found: {json_path}")
            return None
        
        # Load the chunks data
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Retrieve the specific chunk
        if chunk_id in chunks_data:
            return chunks_data[chunk_id]["text"]
        else:
            logger.warning(f"Chunk ID {chunk_id} not found in chunks file")
            return None
    
    except Exception as e:
        logger.error(f"Error loading chunk from JSON: {str(e)}")
        return None
    
def truncate_text(text, max_bytes=1000):
    """Truncate text to ensure it doesn't exceed max_bytes"""
    try:
        encoded = text.encode('utf-8')
        if len(encoded) <= max_bytes:
            return text
        return encoded[:max_bytes].decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error truncating text: {str(e)}")
        return text[:max_bytes // 2]  # Fallback truncation


# Task 3: Generate embeddings
# Update model ID to the correct version
def get_embedding(text, client):
    """Generate embedding using hash-based approach for consistency."""
    try:
        # Use hash-based embedding for consistency
        hash_object = hashlib.sha256(text.encode())
        hash_bytes = hash_object.digest() * 6  # Multiply to get desired length
        embedding = [int(byte)/255.0 for byte in hash_bytes][:192]  # Use 192 dimensions
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

# Task 6: Search Pinecone for relevant chunks
def get_query_embedding(query_text, client):
    """Generate query embedding using the same hash-based approach."""
    try:
        logger.info(f"Generating embedding for query: {query_text[:100]}...")
        # Use the same hash-based embedding approach
        hash_object = hashlib.sha256(query_text.encode())
        hash_bytes = hash_object.digest() * 6  # Multiply to get desired length
        embedding = [int(byte)/255.0 for byte in hash_bytes][:192]  # Use 192 dimensions
        logger.info(f"Generated query embedding with dimension: {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        raise


# Task 7: Generate response using LLM    

def extract_links_from_text(text):
    """Extract URLs from text using regex pattern matching"""
    import re
    # Pattern to match URLs with or without protocol
    url_pattern = r'https?://[^\s)]+|www\.[^\s)]+|(?<=\()http[^\s)]+(?=\))'
    matches = re.findall(url_pattern, text)
    return matches

# Task: Enhanced response generation using LiteLLM
def generate_response(query, context_chunks, client, model_id="claude-3-5-sonnet-20241022", metadata=None):
    """Generate response using Anthropic with context chunks"""
    try:
        logger.info(f"Generating response for query with {len(context_chunks)} context chunks")
        
        # Extract text from chunks
        if isinstance(context_chunks[0], dict):
            context = "\n".join([chunk['metadata']['text_preview'] for chunk in context_chunks])
        else:
            context = "\n".join(context_chunks)
            
        # Create message for Anthropic using the correct format
        response = client.messages.create(
            model=model_id,  # Use the correct model name
            system="You are a helpful assistant. Answer the question based on the provided context.",
            messages=[{
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer the question based on the context provided."
            }],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "content": response.content[0].text,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {"error": str(e)}

def load_data_to_local_db(markdown_content, chunking_strategy, client, file_name=None, namespace=None):
    """Process document and load vectors into local SQLite database."""
    print(f"Processing document with {chunking_strategy} strategy...")
    chunks = process_document_with_chunking(markdown_content, chunking_strategy)
    if not chunks:
        raise ValueError("No chunks were generated from the document")
    print(f"Generated {len(chunks)} chunks from document")

    vectors = []
    for i, chunk in enumerate(chunks):
        text = chunk if isinstance(chunk, str) else chunk.get("text", "")
        vector = get_embedding(text, client)  # Pass the client here
        metadata = {
            "chunk_id": i,
            "file_name": file_name or "Unnamed",
            "namespace": namespace or "default",
            "text_preview": text[:100]
        }
        vectors.append({
            "id": f"{namespace}_{i}" if namespace else f"{file_name}_{i}",
            "values": vector,
            "metadata": metadata
        })

    save_vectors_to_local_db(vectors)
    print(f"Successfully saved {len(vectors)} vectors to local database.") 

def query_local_rag(query, model_id, client, top_k=5, namespace=None):
    """Query local SQLite database for relevant chunks."""
    query_embedding = get_query_embedding(query, client)  # Pass the client here
    results = query_local_db(query_embedding, top_k=top_k)
    if not results:
        return {"answer": "No relevant information found.", "sources": []}

    # Format results
    sources = []
    for result in results:
        vector_id, similarity, metadata = result
        sources.append({
            "score": similarity,
            "document": metadata.get("file_name", "Unknown"),
            "preview": metadata.get("text_preview", "No preview available")
        })

    # Generate response using LiteLLM
    response = generate_response(
        query=query,
        context_chunks=[src["preview"] for src in sources],
        client=client,  # Pass the client here
        model_id=model_id
    )

    return {
        "answer": response["content"],
        "sources": sources
    }
    

# def view_first_10_vectors():
#     """View the first 10 vectors in the Pinecone index"""
#     try:
#         # Get existing connections
#         client, _, index, index_name = get_or_create_connections()
        
#         if index is None:
#             logger.error("No Pinecone index exists. Please create embeddings first.")
#             return
        
#         # Query to get first 10 vectors
#         results = index.query(
#             vector=[0] * 1536,  # Dummy vector
#             top_k=10,
#             include_metadata=True
#         )
        
#         # Get the first 10 items
#         first_10 = results.matches
        
#         print("\nFirst 10 vectors in the index:")
#         for i, match in enumerate(first_10, 1):
#             print(f"\n{i}. Vector ID: {match.id}")
#             print(f"   File: {match.metadata.get('file_name', 'Unknown')}")
#             print(f"   Preview: {match.metadata['text_preview'][:100]}...")
            
#     except Exception as e:
#         print(f"Error fetching vectors: {str(e)}")

# Update the main function call
def main():
    """Main function to run the RAG pipeline"""
    try:
        # Initialize the Anthropic client
        client = initialize_anthropic_client()
        
        # Initialize the local database
        initialize_local_db()
        
        # Define the file path for the markdown file
        file_path = "chromadb_pipeline_filesturcture.md"
        
        # Read the content of the markdown file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        # Load data into the local database
        print(f"Loading data from {file_path} into local database...")
        load_data_to_local_db(
            markdown_content=markdown_content,
            chunking_strategy="recursive_chunking",
            client=client,
            file_name=os.path.basename(file_path)
        )
        
        # Test query
        print("\nTesting query...")
        query = "What is the structure of the ChromaDB pipeline?"
        response = query_local_rag(
            query=query,
            model_id="claude-3-5-sonnet-20241022",  # Corrected model name
            client=client,
            top_k=5
        )
        
        # Print results
        print("\nResults:")
        print(f"Answer: {response['answer']}")
        print("\nSources:")
        for source in response['sources']:
            print(f"- {source['document']}: {source['preview']} (Score: {source['score']:.4f})")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        logger.error(f"Error in main execution: {str(e)}")


def query_local_db(query_vector, top_k=5, db_path="local_vectors.db"):
    """Query vectors from SQLite database using cosine similarity."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, vector, metadata FROM vectors")
    results = []
    for row in cursor.fetchall():
        vector_id, vector_blob, metadata = row
        stored_vector = pickle.loads(vector_blob)  # Deserialize vector
        
        # Ensure the dimensions match
        if len(query_vector) != len(stored_vector):
            logger.error(f"Dimension mismatch: query_vector ({len(query_vector)}) vs stored_vector ({len(stored_vector)})")
            continue
        
        similarity = np.dot(query_vector, stored_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
        )
        results.append((vector_id, similarity, json.loads(metadata)))
    conn.close()
    # Sort results by similarity and return top_k
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    return results


if __name__ == "__main__":
    main()

MODEL_CONFIGS = {
        "claude-3.5-sonnet": {"provider": "anthropic"}
}