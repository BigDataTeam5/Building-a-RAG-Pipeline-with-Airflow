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
            if current_chunk:
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
    if chunking_strategy == "Character-Based Chunking":
        return character_based_chunking(text)
    elif chunking_strategy == "Recursive Character/Token Chunking":
        return recursive_chunking(text)
    elif chunking_strategy == "Semantic Chuking(Kamradt Method)":
        return semantic_chunking(text)
    else:
        # Default to recursive chunking if strategy not recognized
        return recursive_chunking(text)

# Import LiteLLM response generator
from Backend.litellm_query_generator import generate_response, MODEL_CONFIGS
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
        index_name = os.getenv("PINECONE_INDEX_NAME", "pinecone_embeddings")
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
                dimension=1024,
                metric="cosine",  # Default metric, can be overridden at query time
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
def get_embedding(text, client, model="text-embedding-ada-002"):
    """Generate embedding using OpenAI API"""
    try:
        response = client.embeddings.create(
            model=model,
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
            if i % 10 == 0:
                logger.info(f"Processing chunk {i}/{len(chunks)}")
                
            text = chunk.get("text", "")
            filename = chunk.get("file_path", "")
            chunk_id = str(chunk.get("id", str(uuid4())))
            
            try:
                vector = get_embedding(text, client)
                truncated_text = truncate_text(text)
                vectors.append({
                    "id": chunk_id,
                    "values": vector,
                    "metadata": {
                        "text_preview": truncated_text,
                        "file_name": filename,
                        "original_length": len(text)
                    }
                })
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
def get_query_embedding(query_text, client, model="text-embedding-ada-002"):
    """Generate embedding for the query text"""
    try:
        response = client.embeddings.create(
            model=model,
            input=[query_text]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        raise

def search_pinecone(query, client, index, top_k=3, similarity_metric="cosine"):
    """Search Pinecone index for relevant chunks using specified similarity metric"""
    try:
        logger.info(f"Searching for: '{query}' with top_k={top_k} and similarity_metric={similarity_metric}")
        query_embedding = get_query_embedding(query, client)
        
        # Apply similarity metric at query time
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            metric=similarity_metric  # Apply similarity metric at query time
        )
        
        logger.info(f"Found {len(results['matches'])} matches")
        
        # Add similarity metric used to each match's metadata
        for match in results['matches']:
            if 'metadata' in match:
                match['metadata']['similarity_metric_used'] = similarity_metric
        
        return results
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
def enhanced_generate_response(query, context_chunks, client, model_id="gpt4o", metadata=None):
    """Generate response using retrieved chunks with LiteLLM
    
    Args:
        query (str): User query
        context_chunks (list): Retrieved context chunks
        client: OpenAI client (not used with LiteLLM)
        model_id (str): Model ID for LiteLLM (from MODEL_CONFIGS)
        metadata (list): Optional metadata for chunks
        
    Returns:
        dict: Response and usage information
    """
    try:
        logger.info(f"Generating response for query with {len(context_chunks)} context chunks")
        
        # For LiteLLM, we need the text of chunks
        if isinstance(context_chunks[0], dict):
            chunks = [chunk['metadata']['text_preview'] for chunk in context_chunks]
        else:
            chunks = context_chunks
            
        # Use LiteLLM response generator
        llm_response = generate_response(
            chunks=chunks,
            query=query,
            model_id=model_id,
            metadata=metadata
        )
        
        # Format the response to match the expected structure
        return {
            "content": llm_response.get("answer", "Error generating response"),
            "usage": llm_response.get("usage", {})
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

# Task 8: Enhanced Interactive Q&A function
def enhanced_interactive_qa(client, index, query, model_id="gpt4o", chunking_strategy=None, similarity_metric="cosine"):
    """Enhanced Q&A function with LiteLLM and chunking options"""
    logger.info("Processing Q&A request")
    
    try:
        # Search for relevant chunks with specified similarity metric
        results = search_pinecone(query, client, index, similarity_metric=similarity_metric)
        
        if not results['matches']:
            return {"answer": "No relevant information found.", "usage": {}}
        
        # Generate response using chunks with the enhanced function
        response_data = enhanced_generate_response(
            query, 
            results['matches'], 
            client,
            model_id=model_id,
            metadata=[match['metadata'] for match in results['matches']]
        )
        
        # Prepare response with sources and similarity metric used
        sources = []
        for i, match in enumerate(results['matches'], 1):
            sources.append({
                "score": match['score'],
                "file": match['metadata'].get('file_name', 'Unknown'),
                "preview": match['metadata']['text_preview'][:100],
                "similarity_metric": match['metadata'].get('similarity_metric_used', similarity_metric)
            })
        
        return {
            "answer": response_data["content"],
            "usage": response_data["usage"],
            "sources": sources,
            "similarity_metric_used": similarity_metric
        }
            
    except Exception as e:
        logger.error(f"Error in Q&A: {str(e)}")
        raise

def load_data_to_pinecone(json_file_path, chunking_strategy=None):
    """Load data from JSON and upload to Pinecone vector database
    
    Args:
        json_file_path (str): Path to the JSON file containing chunks
        chunking_strategy (str, optional): Chunking strategy to apply
        
    Returns:
        dict: Status information about the upload process
    """
    try:
        # Initialize connections
        client, pc, index, index_name = initialize_connections()
        
        if not json_file_path:
            logger.warning("No JSON file path provided")
            return {"error": "No JSON file path provided"}
            
        # Load chunks and upload to vector database
        chunks = load_chunks_from_json(json_file_path)
        
        # Apply conditional chunking if strategy is specified
        if chunking_strategy:
            logger.info(f"Applying {chunking_strategy} chunking strategy")
            full_text = "\n\n".join([chunk.get("text", "") for chunk in chunks])
            chunks = process_document_with_chunking(full_text, chunking_strategy)
            chunks = [{"text": chunk, "file_path": json_file_path, "id": str(uuid4())} for chunk in chunks]
        
        # Prepare and upload vectors
        vectors = prepare_vectors_for_upload(chunks, client)
        total_uploaded = upload_vectors_to_pinecone(vectors, index)
        
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
        # Initialize connections
        client, pc, index, index_name = initialize_connections()
        
        # Process query with specified similarity metric
        if query:
            return enhanced_interactive_qa(client, index, query, model_id, None, similarity_metric)
        else:
            logger.warning("No query provided")
            return {"error": "No query provided"}
        
    except Exception as e:
        logger.error(f"Error in querying Pinecone RAG: {str(e)}")
        return {"error": str(e)}

# Main execution function with enhanced options
def run_rag_pipeline(json_file_path=None, query="What is Nvidia?", model_id="gpt4o", 
                     chunking_strategy=None):
    """Run the complete RAG pipeline with enhanced options"""
    try:
        result = {}
        
        # Step 1: Load data to Pinecone if a file path is provided
        if json_file_path:
            load_result = load_data_to_pinecone(json_file_path, chunking_strategy)
            if "error" in load_result:
                return load_result
            result["load_status"] = load_result
        
        # Step 2: Query the database with specified similarity metric
        if query:
            query_result = query_pinecone_rag(query, model_id, chunking_strategy)
            if "error" in query_result:
                return query_result
            result.update(query_result)
            return result
        else:
            logger.warning("No query provided")
            return {"error": "No query provided"}
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        return {"error": str(e)}
        
# Entry point
if __name__ == "__main__":
    default_file_path = r"output\kamradt_chunking\chunks.json"
    run_rag_pipeline(default_file_path)