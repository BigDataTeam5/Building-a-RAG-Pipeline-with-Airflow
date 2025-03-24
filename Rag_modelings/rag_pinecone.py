import json
import openai
from uuid import uuid4  
from dotenv import load_dotenv
import os
import sys
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import numpy as np
import tiktoken
import logging

# create chunking functions for each of the chunking strategies
def character_based_chunking(text, chunk_size=400, overlap=50):
    """Simple fixed-size character chunking with overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Move pointer with overlap

    return chunks

def recursive_chunking(text, chunk_size=400, overlap=50):
    """Simple recursive chunking based on delimiters."""
    # First split by double newlines (paragraphs)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(para) > chunk_size:
            # Add current chunk if it's not empty
            if (current_chunk):
                chunks.append(current_chunk)
            
            # If paragraph is longer than chunk size, split it further
            if len(para) > chunk_size:
                sentences = para.split(". ")
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                        if len(sentence) > chunk_size:
                            # Split by spaces if sentence is still too long
                            words = sentence.split(" ")
                            current_chunk = ""
                            for word in words:
                                if len(current_chunk) + len(word) > chunk_size:
                                    chunks.append(current_chunk)
                                    current_chunk = word + " "
                                else:
                                    current_chunk += word + " "
                        else:
                            current_chunk = sentence + ". "
                    else:
                        current_chunk += sentence + ". "
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def semantic_chunking(text, avg_size=300, min_size=50):
    """Simplified semantic chunking."""
    # For simplicity, we'll use recursive chunking as a fallback
    # In a real implementation, this would use semantic embeddings
    return recursive_chunking(text, avg_size, min_size)

def chunk_document(text, chunking_strategy):
    """Chunk document using the specified strategy."""
    if (chunking_strategy == "Character-Based Chunking"):
        return character_based_chunking(text)
    elif (chunking_strategy == "Recursive Character/Token Chunking"):
        return recursive_chunking(text)
    elif (chunking_strategy == "Semantic Chuking(Kamradt Method)"):
        return semantic_chunking(text)
    else:
        # Default to recursive chunking if strategy not recognized
        return recursive_chunking(text)

# Import LiteLLM response generator
sys.path.append("Backend")
from litellm_query_generator import generate_response, MODEL_CONFIGS
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_pipeline")

# Task 0: Implement conditional chunking
def process_document_with_chunking(text, chunking_strategy="Semantic Chuking(Kamradt Method)"):
    """Process document using the specified chunking strategy.
    
    Args:
        text (str): The document text to chunk
        chunking_strategy (str): The chunking strategy to use
            Options: "Character-Based Chunking", "Recursive Character/Token Chunking", "Semantic Chuking(Kamradt Method)"
            
    Returns:
        List[str]: The chunked document
    """
    try:
        logger.info(f"Chunking document using {chunking_strategy} strategy...")
        chunks = chunk_document(text, chunking_strategy)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        raise

# Task 1: Initialize connections and environment
def initialize_connections():
    """Initialize connections to OpenAI and Pinecone"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize API keys
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")
        index_name = os.getenv("PINECONE_INDEX_NAME", "pinecone-embeddings")
        
        # Sanitize index name to conform to Pinecone naming requirements
        # Only allow lowercase alphanumeric characters and hyphens
        import re
        index_name = re.sub(r'[^a-z0-9\-]', '-', index_name.lower())
        logger.info(f"Using Pinecone index name: {index_name}")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Missing required API keys in environment variables")
            
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Create or connect to Pinecone index - using a single index regardless of similarity metric
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # Changed to match text-embedding-ada-002 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region=pinecone_env
                )
            )
        index = pc.Index(index_name)
        
        logger.info("Connections initialized successfully")
        return client, pc, index, index_name
    except Exception as e:
        logger.error(f"Error initializing connections: {str(e)}")
        raise

# Task 2: Load and preprocess data
def load_chunks_from_json(file_path):
    """Load document chunks from JSON file"""
    try:
        logger.info(f"Loading chunks from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        chunks = data.get("chunks", [])
        logger.info(f"Loaded {len(chunks)} chunks from JSON file")
        return chunks
    except Exception as e:
        logger.error(f"Error loading chunks from JSON: {str(e)}")
        raise

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
def get_embedding(text: str, client):
    """Generate embedding using OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def batch_vectors(vectors, batch_size=100):
    """Yield vectors in batches"""
    for i in range(0, len(vectors), batch_size):
        yield vectors[i:i + batch_size]

# Task 4: Prepare vectors for upload
def prepare_vectors_for_upload(chunks, client):
    """Convert chunks to vectors for Pinecone upload"""
    try:
        logger.info(f"Preparing {len(chunks)} chunks for vectorization")
        vectors = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                text = chunk
                chunk_id = str(i)
            else:
                text = chunk.get("text", "")
                chunk_id = str(chunk.get("id", i))
            
            try:
                vector = get_embedding(text, client)
                # Store full text in metadata
                vectors.append({
                    "id": chunk_id,
                    "values": vector,
                    "metadata": {
                        "text_preview": text,  # Store full text
                        "file_name": f"chunk_{chunk_id}",
                        "original_length": len(text)
                    }
                })
                logger.info(f"Processed chunk {chunk_id} with length {len(text)}")
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                continue
                
        logger.info(f"Prepared {len(vectors)} vectors for upload")
        return vectors
    except Exception as e:
        logger.error(f"Error preparing vectors: {str(e)}")
        raise

# Task 5: Upload vectors to Pinecone
def upload_vectors_to_pinecone(vectors, index, batch_size=100):
    """Upload prepared vectors to Pinecone index in batches"""
    try:
        total_uploaded = 0
        batch_count = 0
        
        logger.info(f"Uploading {len(vectors)} vectors to Pinecone in batches of {batch_size}")
        for batch in batch_vectors(vectors, batch_size):
            batch_count += 1
            try:
                index.upsert(batch)
                total_uploaded += len(batch)
                logger.info(f"Batch {batch_count}: Uploaded {len(batch)} vectors. Total: {total_uploaded}/{len(vectors)}")
            except Exception as e:
                logger.error(f"Error uploading batch {batch_count}: {str(e)}")
                continue
                
        logger.info(f"Finished uploading. Total vectors uploaded: {total_uploaded}")
        return total_uploaded
    except Exception as e:
        logger.error(f"Error in upload process: {str(e)}")
        raise

# Task 6: Search Pinecone for relevant chunks
def get_query_embedding(query_text, client):
    """Generate embedding for the query text"""
    try:
        logger.info(f"Generating embedding for query: {query_text[:100]}...")
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[query_text]
        )
        embedding = response.data[0].embedding
        logger.info(f"Generated embedding with dimension: {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        raise

def search_pinecone(query, client, index, top_k=5):
    """Search Pinecone index for relevant chunks using specified similarity metric"""
    try:
        logger.info(f"Searching for: '{query}' with top_k={top_k}")
        
        # Generate query embedding
        query_embedding = get_query_embedding(query, client)
        
        # Debug log the query vector
        logger.info(f"Query embedding dimension: {len(query_embedding)}")
        
        # Search Pinecone with namespace
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=True
        )
        
        # Debug logging for results
        logger.info(f"Raw results matches count: {len(results.matches)}")
        
        # Enhanced logging for debugging
        for match in results.matches:
            logger.info(f"Match ID: {match.id}")
            logger.info(f"Match Score: {match.score}")
            logger.info(f"Match Metadata: {match.metadata}")
        
        # Convert Pinecone response to dictionary format
        formatted_results = {
            'matches': [
                {
                    'id': match.id,
                    'score': float(match.score),
                    'metadata': match.metadata,
                    'text': match.metadata.get('text_preview', '')
                } for match in results.matches
            ]
        }
        
        if not formatted_results['matches']:
            logger.warning("No matches found in search results")
        else:
            logger.info(f"Found {len(formatted_results['matches'])} matches")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching Pinecone: {str(e)}")
        raise

# Task 7: Generate response using LLM
def count_tokens(text, model="gpt-3.5-turbo"):
    """Count the number of tokens in a text string"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        # Rough estimation as fallback
        return len(text) // 4


# Task: Enhanced response generation using LiteLLM
def enhanced_generate_response(query, context_chunks, client, model_id="gpt-3.5-turbo", metadata=None):
    """Generate response using OpenAI with context chunks
    
    Args:
        query (str): User query
        context_chunks (list): Retrieved context chunks
        client: OpenAI client
        model_id (str): Model ID for OpenAI
        metadata (list): Optional metadata for chunks
        
    Returns:
        dict: Response and usage information
    """
    try:
        logger.info(f"Generating response for query with {len(context_chunks)} context chunks")
        
        # Extract text from chunks
        if isinstance(context_chunks[0], dict):
            context = "\n".join([chunk['metadata']['text_preview'] for chunk in context_chunks])
        else:
            context = "\n".join(context_chunks)
            
        # Create messages for chat completion
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question based on the provided context."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer the question based on the context provided."
            }
        ]
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        # Format the response
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {"error": str(e)}

# Task 8: Enhanced Interactive Q&A function
def enhanced_interactive_qa(client, index, query, model_id="gpt4o", similarity_metric="cosine", top_k=5):
    """Enhanced Q&A function with LiteLLM and chunking options"""
    logger.info("Processing Q&A request")
    
    try:
        # Search for relevant chunks with specified similarity metric
        results = search_pinecone(query, client, index, top_k=5)
        
        if not results['matches']:
            return {"answer": "No relevant information found.", "usage": {}}
        
        # Debug the matches
        logger.info(f"Retrieved {len(results['matches'])} matches for query")
        for match in results['matches']:
            logger.info(f"Match score: {match['score']}, Preview: {match['metadata'].get('text_preview', '')[:100]}")
        
        # Enhanced formatting of sources
        sources_for_response = []
        for match in results['matches']:
            metadata = match['metadata']
            sources_for_response.append({
                "score": match['score'],
                "file": metadata.get('file_name', 'Unknown'),
                "preview": metadata.get('text_preview', '')[:150],
                "similarity_metric": metadata.get('similarity_metric_used', similarity_metric)
            })
        
        # Generate response using chunks
        response_data = enhanced_generate_response(
            query, 
            results['matches'], 
            client,
            model_id=model_id,
            metadata=[match['metadata'] for match in results['matches']]
        )
        
        return {
            "answer": response_data["content"],
            "usage": response_data["usage"],
            "sources": sources_for_response,
            "similarity_metric_used": similarity_metric
        }
            
    except Exception as e:
        logger.error(f"Error in Q&A: {str(e)}")
        raise

def load_data_to_pinecone(json_file_path, chunking_strategy=None):
    """Load data from JSON and upload to Pinecone vector database"""
    try:
        # Initialize connections
        client, pc, index, index_name = initialize_connections()
        
        # Load chunks
        logger.info(f"Loading chunks from {json_file_path}")
        chunks = load_chunks_from_json(json_file_path)
        
        # Prepare vectors with enhanced metadata
        vectors = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            vector = get_embedding(text, client)
            
            vectors.append({
                "id": str(chunk.get("id", i)),
                "values": vector,
                "metadata": {
                    "text_preview": text,
                    "file_name": chunk.get("file_path", f"chunk_{i}"),
                    "original_length": len(text),
                    "chunk_index": i
                }
            })
            logger.info(f"Prepared vector {i} with text length {len(text)}")
        
        # Upload vectors in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:min(i + batch_size, len(vectors))]
            index.upsert(vectors=batch)
            total_uploaded += len(batch)
            logger.info(f"Uploaded batch of {len(batch)} vectors. Total: {total_uploaded}")
        
        # Verify upload
        stats = index.describe_index_stats()
        logger.info(f"Index stats after upload: {stats}")
        
        return {
            "status": "success",
            "total_chunks": len(chunks),
            "total_vectors_uploaded": total_uploaded,
            "index_name": index_name
        }
        
    except Exception as e:
        logger.error(f"Error in loading data to Pinecone: {str(e)}")
        return {"error": str(e)}
    
def query_pinecone_rag(query, model_id="gpt4o", similarity_metric="cosine"):
    """Query the Pinecone database and generate a response using RAG with specified similarity metric
    
    Args:
        query (str): User query to process
        model_id (str): Model ID for response generation
        similarity_metric (str, optional): Similarity metric to use for vector search
        
    Returns:
        dict: Response from the RAG system including answer and sources
    """
    try:
        # Initialize connections (only using client and index)
        client, _, index, _ = initialize_connections()
        
        # Process query with specified similarity metric
        if query:
            return enhanced_interactive_qa(client, index, query, model_id, similarity_metric)
        else:
            logger.warning("No query provided")
            return {"error": "No query provided"}
        
    except Exception as e:
        logger.error(f"Error in querying Pinecone RAG: {str(e)}")
        return {"error": str(e)}

# Main execution function with enhanced options
def run_rag_pipeline(query="What is Nvidia?", model_id="gpt-3.5-turbo", similarity_metric="cosine", top_k=5):
    """Query existing Pinecone index and generate RAG response with enhanced formatting"""
    try:
        # Validate parameters
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query parameter")
            return {"error": "Invalid query - must be a non-empty string"}
            
        logger.info(f"Running RAG pipeline with query: '{query}', model: {model_id}")
        
        # Initialize connections
        client, _, index, _ = initialize_connections()
        
        # Search for relevant chunks with correct parameter order
        search_results = search_pinecone(query, client, index, top_k)
        
        if not search_results['matches']:
            logger.warning("No matches found in vector store")
            print("\n" + "="*50)
            print("❌ NO RESULTS FOUND")
            print("="*50 + "\n")
            return {"answer": "No relevant information found in the knowledge base."}
        
        # Generate response using chunks
        response_data = enhanced_generate_response(
            query=query,
            context_chunks=search_results['matches'],
            client=client,
            model_id=model_id
        )
        
        if "error" in response_data:
            logger.error(f"Error in response generation: {response_data['error']}")
            return {"error": response_data["error"]}
        
        # Format and display the results
        print("\n" + "="*50)
        print("🔍 SEARCH QUERY:", query)
        print("="*50)
        print("📝 ANSWER:")
        print(response_data["content"])
        print("\n💡 SOURCES:")
        for i, match in enumerate(search_results['matches'], 1):
            print(f"  {i}. Score: {match['score']:.4f}")
            print(f"     File: {match['metadata'].get('file_name', 'Unknown')}")
            preview = match['metadata'].get('text_preview', '')[:150]
            print(f"     Preview: {preview}...")
            print()
        print("📊 USAGE STATS:")
        print(f"    Tokens: {response_data['usage'].get('total_tokens', 0)}")
        print("="*50 + "\n")
        
        # Return structured response
        return {
            "answer": response_data["content"],
            "sources": search_results['matches'],
            "usage": response_data["usage"],
            "similarity_metric": similarity_metric
        }
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        return {"error": str(e)}

# Add this new function after the chunking strategies
def save_chunks_to_json(chunks, strategy_name):
    """Save chunks to a JSON file"""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.join("output", strategy_name.lower().replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare chunks with IDs and metadata
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "id": i,
                "text": chunk,
                "char_length": len(chunk),
                "file_path": f"{strategy_name}_chunk_{i}"
            })
        
        output_data = {
            "strategy": strategy_name,
            "chunk_count": len(chunks),
            "chunks": chunk_data
        }
        
        # Save to JSON file
        output_file = os.path.join(output_dir, "chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error saving chunks to JSON: {str(e)}")
        raise

def serialize_index_stats(stats):
    """Convert Pinecone index stats to JSON serializable format"""
    return {
        'dimension': stats.dimension,
        'index_fullness': stats.index_fullness,
        'metric': stats.metric,
        'namespaces': {
            ns_name: serialize_namespace_summary(ns_summary)
            for ns_name, ns_summary in stats.namespaces.items()
        },
        'total_vector_count': stats.total_vector_count
    }

def serialize_namespace_summary(namespace_summary):
    """Convert Pinecone namespace summary to JSON serializable format"""
    return {
        'vector_count': namespace_summary.vector_count
    }

# Update the main execution section
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    test_text = """
    NVIDIA Corporation is a technology company with a market value of 100 billion dollars.
    The company specializes in graphics processing units (GPUs) for gaming and professional markets,
    as well as system on chip units (SoCs) for the mobile computing and automotive market.
    """

    try:
        # Initialize connections first
        client, pc, index, index_name = initialize_connections()
        
        # Process document
        logger.info("Processing test document...")
        chunks = process_document_with_chunking(test_text, "Character-Based Chunking")
        logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks to JSON
        json_file_path = save_chunks_to_json(chunks, "test_chunks")
        logger.info(f"Saved chunks to {json_file_path}")
        
        # Upload to Pinecone and wait for indexing
        upload_result = load_data_to_pinecone(json_file_path)
        print("Upload result:", json.dumps(upload_result, indent=2))
        
        # Verify index status
        stats = index.describe_index_stats()
        try:
            serialized_stats = serialize_index_stats(stats)
            print("Index stats:", json.dumps(serialized_stats, indent=2))
        except Exception as e:
            print("Raw index stats:", {
                'dimension': stats.dimension,
                'total_vector_count': stats.total_vector_count,
                'metric': stats.metric
            })
        
        # Test queries
        test_queries = [
            "What is NVIDIA's market value?",
            "What does NVIDIA specialize in?",
        ]
        
        import time
        time.sleep(2)  # Add a small delay to ensure indexing is complete
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            
            # Direct search test with default namespace
            search_results = search_pinecone(query, client, index, top_k=3)
            print(f"Search results: {json.dumps(search_results, indent=2)}")
            
            # Full pipeline test
            response = run_rag_pipeline(
                query=query,
                model_id="gpt-3.5-turbo",
                top_k=3
            )
            print("Response:", json.dumps(response, indent=2))
            
    except Exception as e:
        print(f"Error in test process: {str(e)}")