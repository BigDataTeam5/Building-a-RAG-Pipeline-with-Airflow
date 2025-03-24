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
import hashlib
from pathlib import Path
import nltk
import time
from nltk.corpus import stopwords

import logging
from chunking_evaluation.chunking import (
    FixedTokenChunker,
    RecursiveTokenChunker,
    KamradtModifiedChunker,
    ClusterSemanticChunker,
    LLMSemanticChunker
)
encoding = tiktoken.get_encoding("cl100k_base")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
# load english stopwords from nltk
STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update(['would', 'could', 'should', 'might', 'many', 'much'])

def get_existing_connections():
    """Get connections to existing Pinecone index without recreating it"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize API keys
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")
        index_name = os.getenv("PINECONE_INDEX_NAME", "pinecone-embeddings")
        
        # Sanitize index name
        import re
        index_name = re.sub(r'[^a-z0-9\-]', '-', index_name.lower())
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Missing required API keys in environment variables")
            
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Get existing index
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            logger.warning(f"Index {index_name} does not exist, please create it first")
            # Return None for index to indicate it doesn't exist
            return client, pc, None, index_name
        
        index = pc.Index(index_name)
        logger.info(f"Connected to existing Pinecone index: {index_name}")
        return client, pc, index, index_name
    except Exception as e:
        logger.error(f"Error connecting to existing index: {str(e)}")
        raise


def character_based_chunking(text, chunk_size=400, overlap=50):
    """Character-based chunking using FixedTokenChunker."""
    try:
        # Convert chunk_size from characters to approximate tokens
        token_size = chunk_size // 4  # Rough estimation of chars to tokens
        
        # Use FixedTokenChunker with character mode
        chunker = FixedTokenChunker(
            chunk_size=token_size, 
            chunk_overlap=overlap // 4,
            token_encoding=encoding,
            use_chars=True  # Use character-based chunking
        )
        
        chunks = chunker.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks using character-based chunking")
        return chunks
    except Exception as e:
        logger.error(f"Error in character-based chunking: {str(e)}")
        # Fallback to simple chunking if module fails
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

def recursive_chunking(text, chunk_size=400, overlap=50):
    """Recursive chunking using RecursiveTokenChunker."""
    try:
        # Convert chunk_size from characters to approximate tokens
        token_size = chunk_size // 4  # Rough estimation of chars to tokens
        
        # Use RecursiveTokenChunker with proper parameters
        chunker = RecursiveTokenChunker(
            token_size=token_size,
            token_overlap=overlap // 4,
            token_encoding=encoding  # Try with token_encoding
        )
        
        chunks = chunker.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks using recursive chunking")
        return chunks
    except Exception as e:
        logger.error(f"Error in recursive chunking: {str(e)}")
        # Fallback to simple recursive implementation
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

def semantic_chunking(text, avg_size=300, min_size=50):
    """Semantic chunking using KamradtModifiedChunker."""
    try:
        # Convert sizes from characters to approximate tokens
        token_avg_size = avg_size // 4  # Rough estimation of chars to tokens
        token_min_size = min_size // 4  # Rough estimation of chars to tokens
        
        # Fixed parameter names for KamradtModifiedChunker
        chunker = KamradtModifiedChunker(
            desired_chunk_size=token_avg_size,  # Try with desired_chunk_size
            min_chunk_size=token_min_size,
            token_encoding=encoding  # Use token_encoding param
        )
        
        chunks = chunker.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks using Kamradt semantic chunking")
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        
        # Check if index already exists and delete it if it does
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            logger.info(f"Deleting existing Pinecone index: {index_name}")
            pc.delete_index(index_name)
            logger.info(f"Successfully deleted existing index: {index_name}")
        
        # Create a new Pinecone index
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
            else:
                text = chunk.get("text", "")
                chunk_id = str(chunk.get("id", hashlib.md5(text.encode('utf-8')).hexdigest()))
            
            # Store the full chunk with its ID
            chunks_data[chunk_id] = {
                "text": text,
                "index": i,
                "length": len(text)
            }
        
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

def truncate_text_to_token_limit(text, max_tokens=8000):
    """Truncate text to ensure it doesn't exceed OpenAI's embedding model token limit"""
    try:
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate to max_tokens
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except Exception as e:
        logger.error(f"Error truncating text to token limit: {str(e)}")
        # Fallback to character-based truncation (rough estimate)
        max_chars = max_tokens * 4  # Rough estimate of tokens to chars
        return text[:max_chars]

# Task 3: Generate embeddings
def get_embedding(text: str, client):
    """Generate embedding using OpenAI API with token limit handling"""
    try:
        # Ensure text doesn't exceed token limit
        truncated_text = truncate_text_to_token_limit(text)
        
        if len(truncated_text) < len(text):
            logger.warning(f"Text truncated from {len(text)} chars to {len(truncated_text)} chars for embedding")
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[truncated_text]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def batch_vectors(vectors, batch_size=100):
    """Yield vectors in batches"""
    for i in range(0, len(vectors), batch_size):
        yield vectors[i:i + batch_size]

# Update the prepare_vectors_for_upload function
def prepare_vectors_for_upload(chunks, client, index_name):
    """Convert chunks to vectors for Pinecone upload"""
    try:
        logger.info(f"Preparing {len(chunks)} chunks for vectorization")
        
        # First, save the full chunks to a JSON file
        json_path = save_chunks_to_json(chunks, index_name)
        
        vectors = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                text = chunk
                # Create a consistent ID based on content hash
                chunk_id = hashlib.md5(text.encode('utf-8')).hexdigest()
            else:
                text = chunk.get("text", "")
                chunk_id = str(chunk.get("id", hashlib.md5(text.encode('utf-8')).hexdigest()))
            
            try:
                vector = get_embedding(text, client)
                
                # Store truncated preview in metadata but keep ID for full retrieval
                preview = text[:1000] if len(text) > 1000 else text
                vectors.append({
                    "id": chunk_id,
                    "values": vector,
                    "metadata": {
                        "text_preview": preview,  # Truncated preview
                        "chunk_id": chunk_id,  # ID for retrieval from JSON
                        "file_name": f"chunk_{i}",
                        "original_length": len(text),
                        "full_text_path": json_path  # Path to the JSON file
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
def generate_response(query, context_chunks, client, model_id="gpt-3.5-turbo", metadata=None):
    """Generate response using LiteLLM with context chunks
    
    Args:
        query (str): User query
        context_chunks (list): Retrieved context chunks
        client: OpenAI client (not used with LiteLLM)
        model_id (str): Model ID for response generation
        metadata (list): Optional metadata for chunks
        
    Returns:
        dict: Response and usage information
    """
    try:
        logger.info(f"Generating response for query with {len(context_chunks)} context chunks using LiteLLM")

        # Extract text and prepare chunks for LiteLLM
        chunk_texts = []
        formatted_metadata = []
        
    
        if isinstance(context_chunks[0], dict):
            for chunk in context_chunks:
                # Extract text from chunk
                chunk_texts.append(chunk['metadata']['text_preview'])
                
                # Format metadata as expected by LiteLLM generator
                formatted_metadata.append({
                    "source": chunk['metadata'].get('file_name', 'Unknown source'),
                    "similarity_score": chunk['score'],
                    "preview": chunk['metadata'].get('text_preview', '')[:100]
                })
        else:
            # Simple list of text chunks
            chunk_texts = context_chunks
            # Basic metadata if none provided
            if metadata:
                formatted_metadata = metadata
            else:
                formatted_metadata = [{"source": f"Chunk {i+1}", "similarity_score": 0.0} for i in range(len(chunk_texts))]
        
        # Use LiteLLM's generate_response function
        response = generate_response(
            chunks=chunk_texts,
            query=query,
            model_id=model_id,
            metadata=formatted_metadata
        )
        
        # Format the response to match expected output structure
        return {
            "content": response["answer"],
            "usage": response["usage"],
            "model": response["model"]
        }
    except Exception as e:
        logger.error(f"Error generating response with LiteLLM: {str(e)}")
        return {"error": str(e)}
        
# Task 8: Enhanced Interactive Q&A function
def enhanced_interactive_qa(client, index, query, model_id, similarity_metric="cosine", top_k=5):
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
        response_data = generate_response(
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

# Adding extract keyword function for enhanced metadata and search capabilities
def extract_keywords(text, max_keywords=10):
    """Extract key terms from text for better searchability using NLTK"""
    try:
        # Tokenize, lowercase, remove stopwords and very short words
        words = text.lower().replace('\n', ' ').split()
        words = [w.strip('.,!?()[]{}"\'') for w in words]
        words = [w for w in words if w and w not in STOPWORDS and len(w) > 3]
        
        # Count word frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_counts.most_common(max_keywords)]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []
    

def load_data_to_pinecone(markdown_content, chunking_strategy, file_name=None, namespace=None):
    """
    Comprehensive function to process document and load vectors into Pinecone
    
    Args:
        markdown_content (str): Content to process and embed
        chunking_strategy (str): Strategy for chunking the document
        file_name (str, optional): Name of the source file
        namespace (str, optional): Pinecone namespace to store vectors in
        
    Returns:
        dict: Status of the embedding process with details
    """
    try:
        logger.info(f"Starting process to load data to Pinecone with {chunking_strategy} chunking")
        
        # Step 1: Check for existing connections
        client, _, index, index_name = get_existing_connections()
        
        # If index doesn't exist, create a new one
        if index is None:
            logger.info("No existing index found. Creating new Pinecone index...")
            client, _, index, index_name = initialize_connections()
        
        # Generate namespace from filename if not provided
        if namespace is None and file_name:
            namespace = os.path.splitext(file_name)[0].lower().replace(" ", "_")
            logger.info(f"Using auto-generated namespace: {namespace}")
        elif namespace is None:
            namespace = f"default_{int(time.time())}"
            logger.info(f"No namespace provided, using: {namespace}")
        
        # Step 2: Process document with chunking
        logger.info(f"Processing document with {chunking_strategy} strategy...")
        chunks = process_document_with_chunking(markdown_content, chunking_strategy)
        if not chunks:
            raise ValueError("No chunks were generated from the document")
        logger.info(f"Generated {len(chunks)} chunks from document")
        
        doc_title=None
        if file_name:
            #remove the externsion and replace underscore 
            doc_title = os.path.splitext(file_name)[0].replace("_", " ")
            logger.info(f"Document title extracted from filename: {doc_title}")
            
        # Step 3: Save full chunks to JSON for retrieval during queries
        json_path = save_chunks_to_json(chunks, index_name)
        if not json_path:
            logger.warning("Failed to save chunks to JSON, vector retrieval may be limited")
        
        # Step 3.2 : Parse document structure (basic impl)
        doc_structure  = {"title": doc_title or "Unknown"}
        try:
            headings = []
            for line in markdown_content.split("\n"):
                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    text = line.strip("#")
                    headings.append({"level": level, "text": text})
            if headings:
                doc_structure["headings"] = headings
                doc_structure["toc"] = [h for h in headings if h["level"] <= 2]
                logger.info(f"Extracted document structure with {len(headings)} headings")
        except Exception as e:
            logger.error(f"Error extracting document structure: {str(e)}")
        
        # Step 4: Prepare vectors with enhanced metadata
        logger.info(f"Preparing vectors for {len(chunks)} chunks...")
        vectors = []
        
        # Track processing progress
        successful_embeddings = 0
        failures = 0
        
        for i, chunk in enumerate(chunks):
            # Generate a unique chunk ID based on content hash
            text = chunk if isinstance(chunk, str) else chunk.get("text", "")
            chunk_id = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            # Check chunk size before embedding
            token_count = len(encoding.encode(text))
            if token_count > 8000:
                logger.warning(f"Chunk {i+1} has {token_count} tokens, which exceeds the limit. It will be truncated.")
            
            try:
                # Get embedding for chunk (will be truncated internally if needed)
                vector = get_embedding(text, client)
                successful_embeddings += 1
                
                # Create preview for metadata (truncated version)
                preview = text[:1000] if len(text) > 1000 else text

                # Find most relevant heading for this chunk
                chunk_heading = "Unknown section"
                if "headings" in doc_structure:
                    # Simple heuristic - find the last heading before this chunk
                    for h in doc_structure["headings"]:
                        if h["text"] in text or text.find(h["text"]) < 100:
                            chunk_heading = h["text"]
                            break
                
                # Add vector with metadata
                vectors.append({
                    "id": chunk_id,
                    "values": vector,
                    "metadata": {
                        "text_preview": preview,
                        "chunk_id": chunk_id,
                        "file_name": file_name or f"Unnamed Document {i+1}",
                        "doc_title": doc_title or "Unknown",
                        "section": chunk_heading,
                        "original_length": len(text),
                        "token_count": token_count,
                        "full_text_path": json_path,
                        "chunking_strategy": chunking_strategy,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "keywords": extract_keywords(text),
                        "namespace": namespace  # Include namespace in metadata
                    }
                })
                
                # Log progress regularly
                if (i+1) % 10 == 0 or i+1 == len(chunks):
                    logger.info(f"Processed {i+1}/{len(chunks)} chunks, {successful_embeddings} successful, {failures} failures")
            except Exception as e:
                failures += 1
                logger.error(f"Error processing chunk {i+1}/{len(chunks)}: {str(e)}")
                continue
        
        if failures > 0:
            logger.warning(f"Failed to process {failures} chunks out of {len(chunks)}")
        
        # Step 5: Upload vectors to Pinecone in batches
        if not vectors:
            raise ValueError("No vectors were generated from chunks. Can't continue with upload.")
            
        logger.info(f"Uploading {len(vectors)} vectors to Pinecone in namespace '{namespace}'...")
        batch_size = 100
        total_uploaded = 0
        
        # Batch upload vectors with namespace
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:min(i + batch_size, len(vectors))]
            # Upload with namespace parameter
            index.upsert(
                vectors=batch,
                namespace=namespace
            )
            total_uploaded += len(batch)
            logger.info(f"Uploaded batch {i//batch_size + 1}: {len(batch)} vectors to namespace '{namespace}'. Total: {total_uploaded}/{len(vectors)}")
        
        # Get index stats to verify upload
        stats = index.describe_index_stats()
        logger.info(f"Index now contains {stats.total_vector_count} vectors total")
        
        # Check namespace statistics
        namespace_stats = stats.namespaces.get(namespace, {})
        vector_count = namespace_stats.get("vector_count", 0)
        logger.info(f"Namespace '{namespace}' now contains {vector_count} vectors")
        
        return {
            "status": "success",
            "total_chunks": len(chunks),
            "successful_embeddings": successful_embeddings,
            "failed_embeddings": failures,
            "vectors_uploaded": total_uploaded,
            "index_name": index_name,
            "json_path": json_path,
            "doc_title": doc_title,
            "namespace": namespace
        }
        
    except Exception as e:
        logger.error(f"Error loading data to Pinecone: {str(e)}")
        return {"status": "failed", "error": str(e)}
    
def query_pinecone_rag(query, model_id="gpt4o", similarity_metric="cosine", top_k=5, 
                      filter_criteria=None, namespace=None, search_all_namespaces=False):
    """
    Query the Pinecone database and generate a response using RAG with hybrid search
    
    Args:
        query (str): User query to process
        model_id (str): Model ID for response generation (must be in MODEL_CONFIGS)
        similarity_metric (str): Similarity metric for vector search
        top_k (int): Number of top results to retrieve
        filter_criteria (dict, optional): Metadata filters to narrow search
        namespace (str, optional): Specific namespace to search in
        search_all_namespaces (bool): Whether to search across all namespaces
        
    Returns:
        dict: Response including answer, sources and usage information
    """
    try:
        # Validate model_id against available models in MODEL_CONFIGS
        from Backend.litellm_query_generator import MODEL_CONFIGS
        if model_id not in MODEL_CONFIGS:
            logger.warning(f"Model {model_id} not found in MODEL_CONFIGS. Using default model instead.")
            model_id = next(iter(MODEL_CONFIGS.keys()))
            logger.info(f"Using model: {model_id}")
        else:
            logger.info(f"Processing RAG query: '{query}' using model: {model_id}")
        
        # Step 1: Connect to existing index
        client, _, index, index_name = get_existing_connections()
        
        if index is None:
            logger.error("No Pinecone index exists. Please create embeddings first.")
            return {"error": "No Pinecone index exists. Please create embeddings first."}
        
        # Get list of available namespaces
        stats = index.describe_index_stats()
        available_namespaces = list(stats.namespaces.keys())
        logger.info(f"Available namespaces: {available_namespaces}")
        
        if not available_namespaces:
            logger.warning("No namespaces found in the index")
            return {"error": "No data found in the index. Please create embeddings first."}
        
        # Step 2: Analyze query intention and preprocess
        query_lower = query.lower()
        
        # Detect TOC (table of contents) specific queries
        toc_query = False
        if ("table of contents" in query_lower or "toc" in query_lower or 
            "chapters" in query_lower or "sections" in query_lower):
            toc_query = True
            logger.info("Detected table of contents query")
        
        # Detect specific document/book queries using regex and keyword analysis
        target_document = None
        import re
        
        # Look for quoted document titles like "Kelly Vickey"
        quoted_titles = re.findall(r'"([^"]*)"', query)
        quoted_titles.extend(re.findall(r"'([^']*)'", query))
        
        # Look for document titles after indicator words like "book", "document", etc.
        doc_indicators = ["book", "document", "report", "paper", "titled", "called", "named"]
        for indicator in doc_indicators:
            if indicator in query_lower:
                # Find potential title after the indicator word
                match = re.search(rf"{indicator}\s+(\w+(?:\s+\w+){0,5})", query_lower)
                if match:
                    quoted_titles.append(match.group(1))
        
        # Direct keyword check (e.g., "kelly_vickey") - convert to spaces for matching
        keywords = extract_keywords(query)
        potential_docs = []
        for keyword in keywords:
            if "_" in keyword:
                potential_docs.append(keyword.replace("_", " "))
            # Also check for potential document titles by their length and uniqueness
            if len(keyword) > 5 and keyword not in STOPWORDS:
                potential_docs.append(keyword)
        
        # Combine all potential document titles
        generic_titles = {"document", "report", "paper"}
        all_potential_titles = quoted_titles + potential_docs
        if all_potential_titles:
            # Use the longest potential title as it's likely to be most specific
            target_document = max(all_potential_titles, key=len)
            logger.info(f"Targeting specific document: '{target_document}'")
            
            # If targeting specific document and namespace not specified, try to find matching namespace
            if target_document.lower() in generic_titles:
                target_document = None
            elif target_document and not namespace and not search_all_namespaces:
                target_doc_key = target_document.lower().replace(" ", "_")
                matching_namespaces = [ns for ns in available_namespaces 
                                     if target_doc_key in ns]
                if matching_namespaces:
                    namespace = matching_namespaces[0]
                    logger.info(f"Auto-selected namespace '{namespace}' for document '{target_document}'")
        
        
        # Step 3: Build filter based on query analysis
        combined_filter = {}
        
        # Add document filter if targeting a specific document
        if target_document:
            # Split the target document into words for partial matching
            target_words = target_document.lower().split()
            doc_filter = {
                "$or": [
                    {"doc_title": {"$in": target_words}},
                    {"file_name": {"$in": target_words}},
                    {"keywords": {"$in": target_words}}
                ]
            }
            combined_filter = doc_filter
        
        # Add TOC filter if needed
        if toc_query:
            toc_filter = {
                "$or": [
                    {"section": {"$in": ["table of content", "toc", "content"]}},
                    {"keywords": {"$in": ["toc", "content", "table", "chapter", "section"]}},
                ]
            }
            
            if combined_filter:
                # Combine with existing document filter with AND
                combined_filter = {"$and": [combined_filter, toc_filter]}
            else:
                combined_filter = toc_filter
        
        # Merge with user-provided filter_criteria if any
        if filter_criteria:
            if combined_filter:
                final_filter = {"$and": [combined_filter, filter_criteria]}
            else:
                final_filter = filter_criteria
        else:
            final_filter = combined_filter
        
        logger.info(f"Using filter: {final_filter}")
        
        # Step 4: Perform semantic search with embedding
        # Extract important keywords for hybrid scoring
        search_keywords = [word.lower() for word in query.split() 
                        if len(word) > 3 and word.lower() not in STOPWORDS]
        
        logger.info(f"Using keywords for hybrid search: {search_keywords}")
        
        # Generate query embedding
        query_embedding = get_query_embedding(query, client)
        
        # Search Pinecone - get more results for hybrid reranking
        fetch_k = top_k * 3 if search_keywords else top_k
        
        # Determine which namespaces to search
        if namespace and not search_all_namespaces:
            # Search in specific namespace
            logger.info(f"Searching in specific namespace: {namespace}")
            
            # Perform the vector search with namespace
            results = index.query(
                vector=query_embedding,
                top_k=fetch_k,
                include_metadata=True,
                filter=final_filter if final_filter else None,
                namespace=namespace
            )
            
            logger.info(f"Found {len(results.matches)} matches in namespace '{namespace}'")
            
        else:
            # Search across all namespaces
            logger.info("Searching across all namespaces")
            
            if not available_namespaces:
                logger.warning("No namespaces available to search")
                return {"error": "No namespaces available to search"}
            
            # Multi-namespace search approach: search each namespace separately and combine results
            all_matches = []
            
            for ns in available_namespaces:
                logger.info(f"Searching namespace: {ns}")
                ns_results = index.query(
                    vector=query_embedding,
                    top_k=max(3, fetch_k // len(available_namespaces)),  # At least 3 results per namespace
                    include_metadata=True,
                    filter=final_filter if final_filter else None,
                    namespace=ns
                )
                
                # Add namespace information to metadata
                for match in ns_results.matches:
                    match.metadata["namespace"] = ns
                
                all_matches.extend(ns_results.matches)
                logger.info(f"Found {len(ns_results.matches)} matches in namespace '{ns}'")
            
            # Sort all matches by score (descending)
            all_matches.sort(key=lambda x: x.score, reverse=True)
            
            # Create a new results object with the top matches
            from types import SimpleNamespace
            results = SimpleNamespace()
            results.matches = all_matches[:fetch_k]
            
            logger.info(f"Combined results from all namespaces: {len(results.matches)} matches")
        
        if not results.matches:
            logger.warning("No matches found in search results")
            return {
                "answer": f"I couldn't find any relevant information about {target_document if target_document else 'your query'} in the documents I have access to.",
                "usage": {},
                "sources": []
            }
        
        # Step 5: Hybrid search - rerank results based on keywords and metadata
        processed_matches = []
        for match in results.matches:
            metadata = match.metadata
            text_preview = metadata.get('text_preview', '')
            doc_title = metadata.get('doc_title', '')
            file_name = metadata.get('file_name', '')
            chunk_keywords = metadata.get('keywords', [])
            section = metadata.get('section', '')
            match_namespace = metadata.get('namespace', namespace or 'unknown')
            
            # Base score from vector similarity
            base_score = float(match.score)
            
            # Calculate keyword match score (0-1 range)
            keyword_matches = sum(1 for kw in search_keywords if kw in text_preview.lower())
            keyword_score = min(keyword_matches / max(1, len(search_keywords)), 1.0) * 0.3
            
            # Document title match bonus (0-0.3 range)
            title_score = 0
            if target_document and doc_title:
                if target_document.lower() in doc_title.lower():
                    title_score = 0.3
                elif any(word in doc_title.lower() for word in target_document.lower().split()):
                    title_score = 0.2
            
            # TOC relevance score for table of contents queries (0-0.4 range)
            toc_score = 0
            if toc_query:
                if ("table of content" in text_preview.lower() or 
                    "content" in section.lower() or
                    any(kw in ["toc", "content", "chapter"] for kw in chunk_keywords)):
                    toc_score = 0.4
            
            # Position bonus for early chunks in a document
            position_score = 0
            chunk_index = int(metadata.get('chunk_index', 0))
            total_chunks = int(metadata.get('total_chunks', 1))
            if chunk_index == 0:  # First chunk often has TOC or introduction
                position_score = 0.2
            elif chunk_index < total_chunks * 0.2:  # First 20% of chunks
                position_score = 0.1
            
            # Combined score with weights
            combined_score = base_score + keyword_score + title_score + toc_score + position_score
            
            logger.info(f"Namespace: {match_namespace}, Chunk {chunk_index}: Base: {base_score:.2f}, KW: {keyword_score:.2f}, "
                       f"Title: {title_score:.2f}, TOC: {toc_score:.2f}, Pos: {position_score:.2f}, "
                       f"Total: {combined_score:.2f}")
            
            processed_matches.append({
                'match': match,
                'combined_score': combined_score,
                'keyword_score': keyword_score,
                'toc_score': toc_score,
                'preview': text_preview[:100],
                'namespace': match_namespace
            })
        
        # Sort by combined score and take top_k
        processed_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        top_matches = [item['match'] for item in processed_matches[:top_k]]
        
        logger.info(f"Selected {len(top_matches)} top chunks after hybrid scoring")
        
        # Step 6: Retrieve full chunk content and prepare for the LLM
        chunk_texts = []
        formatted_metadata = []
        
        for match in top_matches:
            # Get chunk ID and JSON path from metadata
            chunk_id = match.metadata.get('chunk_id')
            json_path = match.metadata.get('full_text_path')
            match_namespace = match.metadata.get('namespace', namespace or 'unknown')
            
            logger.info(f"Processing match from namespace '{match_namespace}': {match.id}")
            logger.info(f"  Document: {match.metadata.get('doc_title', 'Unknown')}")
            logger.info(f"  Section: {match.metadata.get('section', 'Unknown')}")
            logger.info(f"  Preview: {match.metadata.get('text_preview', '')[:100]}")
            
            # Retrieve full chunk content
            if json_path and chunk_id:
                # Extract index name from JSON path
                index_name = os.path.basename(json_path).split('_chunks.json')[0]
                
                # Load full chunk content from JSON
                full_text = load_chunks_from_json(chunk_id, index_name)
                if full_text:
                    logger.info(f"Retrieved full content ({len(full_text)} chars)")
                    chunk_texts.append(full_text)
                else:
                    # Fallback to preview if full text not available
                    logger.warning(f"Could not retrieve full content, using preview")
                    chunk_texts.append(match.metadata.get('text_preview', ''))
            else:
                # Use preview if chunk_id or json_path not available
                chunk_texts.append(match.metadata.get('text_preview', ''))
            
            # Format metadata for LLM
            formatted_metadata.append({
                "source": match.metadata.get('doc_title', match.metadata.get('file_name', 'Unknown')),
                "section": match.metadata.get('section', 'General content'),
                "similarity_score": float(match.score),
                "preview": match.metadata.get('text_preview', '')[:100],
                "chunking_strategy": match.metadata.get('chunking_strategy', 'unknown'),
                "chunk_index": match.metadata.get('chunk_index', 0),
                "total_chunks": match.metadata.get('total_chunks', 0),
                "keywords": match.metadata.get('keywords', []),
                "namespace": match_namespace
            })
        
        # Step 7: Enhance the query with context awareness
        enhanced_query = query
        
        # Add document-specific context
        if target_document:
            enhanced_query = f"Based on the content about '{target_document}', please answer: {query}"
        elif "document" in query.lower():
            doc_names = set()
            for match in top_matches:
                doc_title = match.metadata.get('doc_title', 'Unknown')
                if doc_title and doc_title != 'Unknown':
                    doc_names.add(doc_title)
            doc_context = ", ".join(doc_names) if doc_names else "the provided chunks"
            enhanced_query = f"Based on the content from {doc_context}, please answer: {query}"
        
        # Add TOC-specific instructions for table of contents queries
        if toc_query:
            enhanced_query = f"Please extract and format the table of contents or chapter list from the provided context. Query: {query}"
        
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # Step 8: Generate response using LiteLLM with the specified model from MODEL_CONFIGS
        logger.info(f"Generating response with {len(chunk_texts)} chunks using {model_id}")
        
        # Import the response generator dynamically to avoid circular imports
        from Backend.litellm_query_generator import generate_response, MODEL_CONFIGS
        
        # Get model configuration
        model_config = MODEL_CONFIGS.get(model_id, {})
        logger.info(f"Using model config: {model_config}")
        
        response = generate_response(
            chunks=chunk_texts,
            query=enhanced_query,
            model_id=model_id,
            metadata=formatted_metadata
        )
        
        # Step 9: Format the final response with enhanced source information
        sources_for_response = []
        for i, match in enumerate(top_matches):
            metadata = match.metadata
            match_namespace = metadata.get('namespace', namespace or 'unknown')
            sources_for_response.append({
                "score": float(match.score),
                "document": metadata.get('doc_title', 'Unknown'),
                "file": metadata.get('file_name', 'Unknown'),
                "section": metadata.get('section', 'Unknown'),
                "preview": metadata.get('text_preview', '')[:150],
                "chunk_index": metadata.get('chunk_index', 0),
                "total_chunks": metadata.get('total_chunks', 0),
                "namespace": match_namespace
            })
        
        return {
            "answer": response["answer"],
            "usage": response["usage"],
            "sources": sources_for_response,
            "similarity_metric_used": similarity_metric,
            "model": response["model"],
            "target_document": target_document,
            "is_toc_query": toc_query,
            "namespaces_searched": [namespace] if namespace else available_namespaces
        }
            
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        return {"error": str(e)}

def view_vectors_by_namespace():
    """View vectors in the Pinecone index organized by namespace"""
    try:
        # Get existing connections
        client, _, index, index_name = get_existing_connections()
        
        if index is None:
            logger.error("No Pinecone index exists. Please create embeddings first.")
            return
        
        # Get stats with namespaces
        stats = index.describe_index_stats()
        namespaces = stats.namespaces
        
        if not namespaces:
            print("No namespaces found in the index.")
            return
        
        print(f"\nIndex '{index_name}' contains {stats.total_vector_count} total vectors")
        print(f"Found {len(namespaces)} namespaces:")
        
        # List all namespaces and their vector counts
        for ns_name, ns_data in namespaces.items():
            print(f"\n📂 Namespace: {ns_name}")
            print(f"   Vectors: {ns_data.vector_count}")
            
            # Get sample vectors from each namespace
            results = index.query(
                vector=[0] * 1536,  # Dummy vector
                top_k=3,            # Just get 3 samples
                include_metadata=True,
                namespace=ns_name
            )
            
            # Show samples
            if results.matches:
                print(f"   Sample vectors:")
                for i, match in enumerate(results.matches, 1):
                    print(f"     {i}. ID: {match.id}")
                    print(f"        Document: {match.metadata.get('doc_title', 'Unknown')}")
                    print(f"        Section: {match.metadata.get('section', 'Unknown')}")
                    print(f"        Preview: {match.metadata.get('text_preview', '')[:80]}...")
            else:
                print("   No vectors found in this namespace.")
                
    except Exception as e:
        print(f"Error viewing vectors by namespace: {str(e)}")

# You can add this to the __main__ section:
# view_vectors_by_namespace()    

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

def view_first_10_vectors():
    """View the first 10 vectors in the Pinecone index"""
    try:
        # Get existing connections
        client, _, index, index_name = get_existing_connections()
        
        if index is None:
            logger.error("No Pinecone index exists. Please create embeddings first.")
            return
        
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

# Update the main execution section
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("rag_pipeline")
    
    def test_pipeline():
                # Add this in the test_pipeline function
        print("\n" + "="*80)
        print("NAMESPACE MANAGEMENT".center(80))
        print("="*80 + "\n")

        print("1. View vectors by namespace")
        print("2. Load document to specific namespace")
        print("3. Query from specific namespace")
        print("4. Skip namespace operations")

        ns_choice = input("\nSelect namespace operation (or press Enter to skip): ").strip()

        if ns_choice == "1":
            view_vectors_by_namespace()
        elif ns_choice == "2":
            # Get namespace name
            namespace = input("Enter namespace name: ").strip()
            if not namespace:
                namespace = f"custom_{int(time.time())}"
                print(f"Using generated namespace: {namespace}")
            
            # Get document path
            doc_path = input("Enter document path (or press Enter for test_document.md): ").strip()
            if not doc_path:
                doc_path = "test_document.md"
            
            # Check if file exists or create it
            if not os.path.exists(doc_path):
                print(f"File not found: {doc_path}")
                # Create a simple test document
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write("# Test Document\n\n## Table of Contents\n\n- Chapter 1: Introduction\n- Chapter 2: Test Content\n\n## Chapter 1\nThis is a test document.")
                print(f"Created test document: {doc_path}")
            
            # Load the document
            with open(doc_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Extract file name from path
            file_name = os.path.basename(doc_path)
            
            # Load data to Pinecone with specified namespace
            print(f"\nLoading {file_name} to namespace '{namespace}'...")
            result = load_data_to_pinecone(
                markdown_content, 
                "recursive_chunking", 
                file_name=os.path.splitext(file_name)[0],
                namespace=namespace
            )
            
            if result["status"] == "success":
                print(f"✅ Successfully loaded document to namespace '{namespace}'!")
                print(f"   - Total chunks: {result['total_chunks']}")
                print(f"   - Vectors uploaded: {result['vectors_uploaded']}")
            else:
                print(f"❌ Failed to load document: {result.get('error', 'Unknown error')}")
                
        elif ns_choice == "3":
            # Get namespace name
            namespace = input("Enter namespace to search (leave empty to search all): ").strip()
            
            # Get query
            user_query = input("Enter your query: ").strip()
            if not user_query:
                user_query = "What is this document about?"
            
            # Get model
            model = input("Select model (gemini, gpt-3.5-turbo, gpt-4o, or press Enter for default): ").strip()
            if not model:
                model = "gpt-3.5-turbo"
            
            # Search with namespace
            print(f"\nSearching {'namespace ' + namespace if namespace else 'all namespaces'} for: '{user_query}'")
            response = query_pinecone_rag(
                query=user_query,
                model_id=model,
                similarity_metric="cosine",
                top_k=5,
                namespace=namespace,
                search_all_namespaces=not namespace
            )
            
            print("\n" + "-"*80)
            if "error" not in response:
                print(f"📝 ANSWER:\n{response['answer']}")
                print("\n🔍 SOURCES:")
                for i, source in enumerate(response.get('sources', []), 1):
                    print(f"{i}. Document: {source.get('document', 'Unknown')}")
                    print(f"   Namespace: {source.get('namespace', 'Unknown')}")
                    print(f"   Section: {source.get('section', 'Unknown')}")
                    print(f"   Score: {source.get('score', 0):.4f}")
                    print(f"   Preview: {source.get('preview', '')[:100]}...")
            else:
                print(f"❌ Query failed: {response['error']}")
            print("-"*80 + "\n")
    test_pipeline()