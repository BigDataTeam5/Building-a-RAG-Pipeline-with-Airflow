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



def character_based_chunking(text, chunk_size=400, overlap=50):
    """Character-based chunking using FixedTokenChunker."""
    try:
        # Convert chunk_size from characters to approximate tokens
        token_size = chunk_size // 4  # Rough estimation of chars to tokens
        
        # Use FixedTokenChunker with correct parameters
        chunker = FixedTokenChunker(
            chunk_size=token_size, 
            chunk_overlap=overlap // 4,
            length_function=lambda text: len(encoding.encode(text)),
            use_chars=True  # Use character-based chunking
        )
        
        chunks = chunker.split_text(text)  # Changed from chunk_text to split_text
        logger.info(f"Created {len(chunks)} chunks using character-based chunking")
        return chunks
    except Exception as e:
        logger.error(f"Error in character-based chunking: {str(e)}")
        # Fallback to simple chunking if module fails
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
def get_or_create_connections(similarity_metric="cosine", file_name=None, namespace=None):
    import re
    """Get existing connections or create new ones based on similarity metric"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize API keys
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Missing required API keys in environment variables")
        similarity_metric  = similarity_metric.replace("_", " ").lower()
        
        # Set index name based on similarity metric and sanitize it
        if "cosine" in similarity_metric:
            index_name = "embedding-cosine"  # Use hyphen instead of underscore
        elif "euclidean" in similarity_metric:
            index_name = "embedding-euclidean"  # Use hyphen instead of underscore
        elif "dot" in similarity_metric:
            index_name = "embedding-dot"  # Use hyphen instead of underscore
        else:
            index_name = "embedding-cosine"  # Use hyphen instead of underscore
            logger.warning(f"Unknown similarity metric '{similarity_metric}', using default 'cosine'")
        
        # Ensure index name follows Pinecone's naming convention (lowercase alphanumeric with hyphens)
        index_name = index_name.lower().replace('_', '-')
        
        logger.info(f"Using Pinecone index: {index_name} with {similarity_metric} similarity")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index already exists
        existing_indexes = pc.list_indexes().names()
        
        # Create the index if it doesn't exist
        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {index_name}")
            
            # Set metric based on similarity_metric
            if "cosine" in similarity_metric:
                metric = "cosine"
            elif "euclidean" in similarity_metric:
                metric = "euclidean"
            elif "dot" in similarity_metric:
                metric = "dotproduct"
            else:
                metric = "cosine"
                
            pc.create_index(
                name=index_name,
                dimension=1536,  # Matches text-embedding-ada-002 dimension
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region=pinecone_env
                )
            )
            logger.info(f"Successfully created new index: {index_name}")
        else:
            logger.info(f"Connected to existing Pinecone index: {index_name}")
        
        # Get the index
        index = pc.Index(index_name)
        
        # Generate namespace from filename if not provided
        if namespace is None and file_name:
            # Sanitize namespace too - only lowercase alphanumeric and hyphens
            namespace = re.sub(r'[^a-z0-9-]', '-', os.path.splitext(file_name)[0].lower())
            logger.info(f"Using auto-generated namespace: {namespace}")
        elif namespace is None:
            namespace = f"default-{int(time.time())}"
            logger.info(f"No namespace provided, using: {namespace}")
        else:
            # Ensure namespace follows naming convention
            namespace = re.sub(r'[^a-z0-9-]', '-', namespace.lower())
        
        logger.info("Connections initialized successfully")
        return client, pc, index, index_name, namespace
        
    except Exception as e:
        logger.error(f"Error initializing connections: {str(e)}")
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
    

def extract_links_from_text(text):
    """Extract URLs from text using regex pattern matching"""
    import re
    # Pattern to match URLs with or without protocol
    url_pattern = r'https?://[^\s)]+|www\.[^\s)]+|(?<=\()http[^\s)]+(?=\))'
    matches = re.findall(url_pattern, text)
    return matches

# Task: Enhanced response generation using LiteLLM
def generate_response(query, context_chunks, client, model_id="gpt-3.5-turbo", metadata=None):
    """Generate response using LiteLLM with context chunks"""
    try:
        logger.info(f"Generating response for query with {len(context_chunks)} context chunks using LiteLLM")

        # Check if this is a link extraction request
        query_lower = query.lower()
        is_link_request = any(term in query_lower for term in ["links", "urls", "link", "url"]) and \
                         any(term in query_lower for term in ["list", "show", "extract", "get", "what", "all"])
        
        # For link extraction requests, directly parse the chunks
        if is_link_request:
            all_links = []
            source_texts = []
            
            # Extract text from chunks
            if isinstance(context_chunks[0], dict):
                for chunk in context_chunks:
                    chunk_text = chunk.get('text', '') or chunk.get('metadata', {}).get('text_preview', '')
                    source_texts.append(chunk_text)
            else:
                source_texts = context_chunks
            
            # Extract links from each chunk
            for text in source_texts:
                links = extract_links_from_text(text)
                all_links.extend(links)
            
            # Remove duplicates while preserving order
            unique_links = []
            for link in all_links:
                if link not in unique_links:
                    unique_links.append(link)
            
            # Format response
            if unique_links:
                link_list = "\n".join([f"- {link}" for link in unique_links])
                response_text = f"Here are all the links found in the NVIDIA document:\n\n{link_list}"
            else:
                response_text = "No links were found in the retrieved document chunks."
            
            # Return formatted response
            return {
                "content": response_text,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": model_id
            }
        
        # For non-link requests, use standard LiteLLM response generation
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
        return {"error": str(e), "content": str(e)}
            


def load_data_to_pinecone(markdown_content, chunking_strategy, file_name=None, namespace=None, similarity_metric="cosine_similarity"):
    """
    Optimized function to process document and load vectors into Pinecone
    with minimal metadata storage
    """
    try:
        # Step 0: Get existing connections and setup
        logger.info(f"Starting process to load data to Pinecone with {chunking_strategy} chunking")
        similarity_metric = similarity_metric.replace("_", " ").lower()
        client, _, index, index_name, namespace = get_or_create_connections(
            similarity_metric=similarity_metric,
            file_name=file_name,
            namespace=namespace
        )

        # Log information about connections
        logger.info(f"Connected to Pinecone index: {index_name}")
        if namespace:
            logger.info(f"Using namespace: {namespace}")
        else:
            logger.warning("No namespace specified, using default")

        # Verify file_name is available
        if file_name:
            logger.info(f"Processing document: {file_name}")
        else:
            logger.info("No file name provided, using unnamed document")
            
        # Step 2: Process document with chunking (optimized to handle base64 images)
        logger.info(f"Processing document with {chunking_strategy} strategy...")
        chunks = process_document_with_chunking(markdown_content, chunking_strategy)
        if not chunks:
            raise ValueError("No chunks were generated from the document")
        logger.info(f"Generated {len(chunks)} chunks from document")
        
        doc_title = None
        if file_name:
            # Remove the extension and replace underscore 
            doc_title = os.path.splitext(file_name)[0].replace("_", " ")
            logger.info(f"Document title extracted from filename: {doc_title}")
            
        # Step 3: Save full chunks to JSON for retrieval during queries
        json_path = save_chunks_to_json(chunks, index_name)
        if not json_path:
            logger.warning("Failed to save chunks to JSON, vector retrieval may be limited")
        
        # Step 3.2: Parse document structure (basic impl)
        doc_structure = {"title": doc_title or "Unknown"}
        try:
            headings = []
            for line in markdown_content.split("\n"):
                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    text = line.strip("#").strip()
                    headings.append({"level": level, "text": text})
            if headings:
                doc_structure["headings"] = headings
                doc_structure["toc"] = [h for h in headings if h["level"] <= 2]
                logger.info(f"Extracted document structure with {len(headings)} headings")
        except Exception as e:
            logger.error(f"Error extracting document structure: {str(e)}")
        
        # Step 4: Prepare vectors with MINIMAL metadata
        logger.info(f"Preparing vectors for {len(chunks)} chunks...")
        vectors = []
        
        # Track processing progress
        successful_embeddings = 0
        failures = 0
        
        for i, chunk in enumerate(chunks):
            # Generate a unique chunk ID based on content hash
            text = chunk if isinstance(chunk, str) else chunk.get("text", "")
            chunk_id = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            # Check if this chunk contains a base64 image
            has_image = "base64" in text and "![" in text
            
            # Skip token counting for base64 images
            if not has_image:
                # Check chunk size before embedding
                token_count = len(encoding.encode(text))
                if token_count > 8000:
                    logger.warning(f"Chunk {i+1} has {token_count} tokens, which exceeds the limit. It will be truncated.")
                    text = truncate_text_to_token_limit(text, 8000)
            else:
                logger.info(f"Chunk {i+1} contains base64 image(s), preserving as-is")
            
            try:
                # Get embedding for chunk
                vector = get_embedding(text, client)
                successful_embeddings += 1
                
                # Find most relevant heading for this chunk
                chunk_heading = "Unknown section"
                if "headings" in doc_structure:
                    # Simple heuristic - find the last heading before this chunk
                    for h in doc_structure["headings"]:
                        if h["text"] in text or text.find(h["text"]) < 100:
                            chunk_heading = h["text"]
                            break
                
                
                metadata = {
                    "chunk_id": chunk_id,  # Essential for retrieving from JSON
                    "file_name": file_name or f"Unnamed-{i}",
                    "doc_title": doc_title or "Unknown",
                    "section": chunk_heading,
                    "has_image": has_image,  # Flag if chunk contains base64 images
                    "chunks_path": json_path  # Path to retrieve full content
                }
                
                # Add vector with minimal metadata
                vectors.append({
                    "id": chunk_id,
                    "values": vector,
                    "metadata": metadata
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
        
        time.sleep(5)
        try:
            # Get index stats to verify upload
            stats = index.describe_index_stats()
            logger.info(f"Index now contains {stats.total_vector_count} vectors total")

            if namespace in stats.namespaces:
                vector_count = stats.namespaces[namespace].vector_count
                logger.info(f"Namespace '{namespace}' now contains {vector_count} vectors")
            else:
                logger.warning(f"Namespace '{namespace}' not found in index stats")
        except Exception as e:
            logger.warning(f"Could not retrieve final stats: {str(e)}")
        
        return {
            "status": "success",
            "total_chunks": len(chunks),
            "successful_embeddings": successful_embeddings,
            "vectors_uploaded": total_uploaded,
            "index_name": index_name,
            "json_path": json_path,
            "doc_title": doc_title,
            "namespace": namespace
        }
        
    except Exception as e:
        logger.error(f"Error loading data to Pinecone: {str(e)}")
        return {"status": "failed", "error": str(e)}

def query_pinecone_rag(query, model_id, similarity_metric="cosine", top_k=5, 
                      namespace=None,json_path=None):
    """
    Optimized query function for Pinecone RAG that reliably retrieves content from JSON files
    """
    try:
        # Validate model_id against available models
        from Backend.litellm_query_generator import MODEL_CONFIGS
        if (model_id not in MODEL_CONFIGS):
            logger.warning(f"Model {model_id} not found in MODEL_CONFIGS. Using default model instead.")
            model_id = next(iter(MODEL_CONFIGS.keys()))
            logger.info(f"Using model: {model_id}")
        else:
            logger.info(f"Processing RAG query: '{query}' using model: {model_id}")
        
        # Step 1: Connect to existing index
        client, _, index, index_name, current_namespace = get_or_create_connections(
            similarity_metric=similarity_metric,
            namespace=namespace
        )
        
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
        
        if namespace and namespace not in available_namespaces:
            logger.warning(f"Namespace '{namespace}' not found. Searching all namespaces instead.")
            namespace = None
        
        # Generate query embedding
        query_embedding = get_query_embedding(query, client)
        
        # Simplified search approach
        if namespace:
            # Search in specific namespace
            logger.info(f"Searching in specific namespace: {namespace}")
            
            # Perform the vector search with namespace
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            logger.info(f"Found {len(results.matches)} matches in namespace '{namespace}'")
            
        else:
            # Search across all namespaces
            logger.info("Searching across all namespaces")
            
            # Multi-namespace search approach: search each namespace separately and combine results
            all_matches = []
            
            for ns in available_namespaces:
                logger.info(f"Searching namespace: {ns}")
                ns_results = index.query(
                    vector=query_embedding,
                    top_k=max(3, top_k // len(available_namespaces)),  # At least 3 results per namespace
                    include_metadata=True,
                    namespace=ns
                )
                
                # Add namespace information to metadata
                for match in ns_results.matches:
                    if not hasattr(match.metadata, 'namespace'):
                        match.metadata["namespace"] = ns
                
                all_matches.extend(ns_results.matches)
                logger.info(f"Found {len(ns_results.matches)} matches in namespace '{ns}'")
            
            # Sort all matches by score (descending) and take top_k
            all_matches.sort(key=lambda x: x.score, reverse=True)
            
            # Create a new results object with the top matches
            from types import SimpleNamespace
            results = SimpleNamespace()
            results.matches = all_matches[:top_k]
            
            logger.info(f"Combined results from all namespaces: {len(results.matches)} matches")
        
        if not results.matches:
            logger.warning("No matches found in search results")
            return {
                "answer": "I couldn't find any relevant information about your query in the documents I have access to.",
                "usage": {},
                "sources": []
            }
        
        # Simplified chunk retrieval from JSON file specified by json_path parameter
        chunk_texts = []
        formatted_metadata = []
        chunk_images = []  # List to collect images
        
        # Log the matches for debugging
        logger.info(f"Retrieved {len(results.matches)} matches for query")
        chunks_dir = Path("chunk_storage")
        
        # Use json_path if provided, otherwise construct the default path
        if json_path:
            json_file_path = Path(json_path)
            logger.info(f"Using provided JSON path: {json_file_path}")
        else:
            json_file_path = chunks_dir / f"{index_name}_chunks.json"
            logger.info(f"Using default JSON path: {json_file_path}")
            
        chunks_data = {}
        if json_file_path.exists():
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                logger.info(f"Loaded {len(chunks_data)} chunks from {json_file_path}")
            except Exception as e:
                logger.error(f"Error loading chunks from {json_file_path}: {str(e)}")
                return {
                    "answer": "I couldn't access the document content needed to answer your question.",
                    "usage": {},
                    "sources": []
                }
        else:
            logger.error(f"Chunk file not found at: {json_file_path}")
            return {
            "answer": "I couldn't find the document content at the specified path.",
            "usage": {},
            "sources": []
            }
        
        # Process each match
        for i, match in enumerate(results.matches):
            match_id = match.id
            match_namespace = match.metadata.get('namespace', namespace or 'unknown')
            
            logger.info(f"Processing match {i+1}: ID={match_id}, Score={match.score}")
            
            # Get chunk text from the loaded JSON data
            if match_id in chunks_data:
                chunk_data = chunks_data[match_id]
                chunk_text = chunk_data.get("text", "")
                logger.info(f"Found chunk {match_id} in JSON file")
                
                # Check for images
                if "images" in chunk_data and chunk_data["images"]:
                    for img in chunk_data["images"]:
                        chunk_images.append(img)
                        logger.info(f"Found {len(chunk_data['images'])} images in chunk")
            
            
                # Add chunk text to the list
                chunk_texts.append(chunk_text)
                
                # Format metadata for LLM
                formatted_metadata.append({
                "source": match.metadata.get('doc_title', match.metadata.get('file_name', 'Unknown')),
                "section": match.metadata.get('section', 'General content'),
                "similarity_score": float(match.score),
                "namespace": match_namespace,
                "has_image": match.metadata.get('has_image', False)
                })
            else:
                # If chunk not found in the file, skip this match
                    logger.warning(f"Chunk {match_id} not found in {json_file_path}, skipping")
                    continue
        
        # Verify we have chunks to process
        if not chunk_texts:
            logger.error("No chunk texts could be retrieved. Query cannot proceed.")
            return {
            "answer": "I couldn't retrieve any document content for the current similarity metric.",
            "usage": {},
            "sources": []
            }
        # Log final counts
        logger.info(f"Successfully retrieved {len(chunk_texts)} chunks")
        logger.info(f"Found {len(chunk_images)} images to include in the response")
        
        # Generate response using LiteLLM
        from Backend.litellm_query_generator import generate_response
        
        # Call with images if available
        if chunk_images:
            logger.info(f"Generating response with {len(chunk_texts)} chunks and {len(chunk_images)} images using {model_id}")
            response = generate_response(
                chunks=chunk_texts,
                query=query,
                model_id=model_id,
                metadata=formatted_metadata,
                images=chunk_images  # Pass the collected images
            )
        else:
            logger.info(f"Generating response with {len(chunk_texts)} chunks using {model_id} (no images)")
            response = generate_response(
                chunks=chunk_texts,
                query=query,
                model_id=model_id,
                metadata=formatted_metadata
            )
        
        # Format sources for response
        sources_for_response = []
        for i, match in enumerate(results.matches):
            metadata = match.metadata
            match_namespace = metadata.get('namespace', namespace or 'unknown')
            
            # Create preview from chunk text
            preview = ""
            if i < len(chunk_texts):
                preview = chunk_texts[i][:150] + "..." if len(chunk_texts[i]) > 150 else chunk_texts[i]
            
            sources_for_response.append({
                "score": float(match.score),
                "document": metadata.get('doc_title', 'Unknown'),
                "file": metadata.get('file_name', 'Unknown'),
                "section": metadata.get('section', 'Unknown'),
                "preview": preview,
                "namespace": match_namespace,
                "has_image": metadata.get('has_image', False)
            })

        return {
            "answer": response["answer"],
            "usage": response["usage"],
            "sources": sources_for_response,
            "similarity_metric_used": similarity_metric,
            "model": response["model"],
            "namespaces_searched": [namespace] if namespace else available_namespaces,
            "images_included": len(chunk_images) > 0
        }
            
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        return {"error": str(e)}
    
    
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

# Update the main execution section

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("rag_pipeline")
    
    def test_pipeline():
        # Add this in the test_pipeline function
        print("\n" + "="*80)
        print("RAG PIPELINE TESTING".center(80))
        print("="*80 + "\n")

        # Add index selection first
        print("Select index to use:")
        print("1. Cosine similarity (embedding-cosine)")
        print("2. Euclidean distance (embedding-euclidean)")
        print("3. Dot product (embedding-dot)")
        
        index_choice = input("\nSelect index (1-3, or press Enter for default cosine): ").strip()
        
        # Set similarity metric based on choice
        if index_choice == "2":
            similarity_metric = "euclidean"
        elif index_choice == "3":
            similarity_metric = "dot"
        else:
            similarity_metric = "cosine"  # Default
        
        print(f"Using index with {similarity_metric} similarity metric\n")
        
        print("NAMESPACE MANAGEMENT".center(80))
        print("="*80 + "\n")

        print("1. View vectors by namespace")
        print("2. Load document to specific namespace")
        print("3. Query from specific namespace")
        print("4. Skip namespace operations")

        ns_choice = input("\nSelect namespace operation (or press Enter to skip): ").strip()

        if ns_choice == "1":
            pass
            # view_vectors_by_namespace()
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
            
            # Load data to Pinecone with specified namespace and similarity metric
            print(f"\nLoading {file_name} to namespace '{namespace}' with {similarity_metric} similarity...")
            result = load_data_to_pinecone(
                markdown_content, 
                "recursive_chunking", 
                file_name=os.path.splitext(file_name)[0],
                namespace=namespace,
                similarity_metric=similarity_metric
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
            model = input("Select model (gemini, gpt-3.5-turbo, claude, or press Enter for default): ").strip()
            if not model:
                model = "gpt-3.5-turbo"
            
            # Search with namespace and selected similarity metric
            print(f"\nSearching {'namespace ' + namespace if namespace else 'all namespaces'} with {similarity_metric} index for: '{user_query}'")
            response = query_pinecone_rag(
                query=user_query,
                model_id=model,
                similarity_metric=similarity_metric,
                top_k=5,
                namespace=namespace
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