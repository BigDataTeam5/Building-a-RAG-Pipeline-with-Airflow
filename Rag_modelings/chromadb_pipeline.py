import os
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from chunking_evaluation.chunking import RecursiveTokenChunker, KamradtModifiedChunker
from chunking_evaluation.utils import openai_token_count
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import shutil
import sys
import nltk
import time

# from chunking_markdowns import chunk_document

# Add parent directory to path to import litellm_query_generator
from Backend.litellm_query_generator import generate_response, MODEL_CONFIGS

# Download NLTK data for sentence tokenization if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ChromaDB settings
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "chromadb_embeddings"

# Embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2"
}

# Initialize sentence transformer model (default)
default_model = SentenceTransformer(EMBEDDING_MODELS["all-MiniLM-L6-v2"])

# Chunking settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def character_based_chunking(text: str, chunk_size: int = CHUNK_SIZE, 
                            overlap: int = CHUNK_OVERLAP) -> List[str]:
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

def recursive_chunking(text: str, chunk_size: int = CHUNK_SIZE, 
                      overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Recursive chunking using RecursiveTokenChunker."""
    chunker = RecursiveTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return chunker.split_text(text)

def semantic_chunking(text: str, embedding_model=None, 
                     avg_size: int = 300, min_size: int = 50) -> List[str]:
    """Semantic chunking using KamradtModifiedChunker."""
    if embedding_model is None:
        embedding_model = default_model
        
    def embedding_function(texts):
        return [embedding_model.encode(text).tolist() for text in texts]
    
    kamradt_chunker = KamradtModifiedChunker(
        avg_chunk_size=avg_size,
        min_chunk_size=min_size,
        embedding_function=embedding_function
    )
    
    return kamradt_chunker.split_text(text)
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



def create_or_clear_collection(client, collection_name: str, embedding_dim: int) -> Any:
    """Create or clear a ChromaDB collection."""
    # Try to delete existing collection
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection {collection_name} does not exist yet: {e}")
    
    # Create new collection
    metadata = {
        "description": "NVIDIA markdown documents",
        "last_refresh": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    collection = client.create_collection(
        name=collection_name,
        metadata=metadata
    )
    return collection

def store_chunks_in_chromadb(
    chunks: List[str], 
    source_info: Dict[str, Any],
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[str, int]:
    """
    Store chunks in ChromaDB.
    
    Args:
        chunks: List of text chunks
        source_info: Dictionary with source information
        embedding_model_name: Name of embedding model to use
        
    Returns:
        Tuple of collection name and number of chunks stored
    """
    # Initialize embedding model
    model = SentenceTransformer(EMBEDDING_MODELS[embedding_model_name])
    
    # Generate embeddings
    embeddings = [model.encode(chunk).tolist() for chunk in chunks]
    embedding_dim = len(embeddings[0])
    
    # Set up ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Use a single collection
    collection_name = COLLECTION_NAME
    
    # Create or clear the collection
    collection = create_or_clear_collection(client, collection_name, embedding_dim)
    
    # Generate IDs and metadata
    file_name = source_info.get("file_name", "unknown")
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": source_info.get("file_path", "unknown"),
            "file_name": file_name,
            "chunk_index": i,
            "year": source_info.get("year", "unknown"),
            "quarter": source_info.get("quarter", "unknown"),
            "embedding_model": embedding_model_name
        }
        for i in range(len(chunks))
    ]
    
    # Add chunks to collection
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Added {len(chunks)} chunks to collection {collection_name}")
    return collection_name, len(chunks)

def retrieve_relevant_chunks(
    query: str,
    similarity_metric: str = "cosine",
    embedding_model_name: str = "all-MiniLM-L6-v2", 
    top_k: int = 5
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Retrieve relevant chunks based on query.
    """
    # Initialize embedding model
    model = SentenceTransformer(EMBEDDING_MODELS[embedding_model_name])
    
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    # Set up ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        # Get collection
        collection = client.get_collection(name=COLLECTION_NAME)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        # Add distance score to metadata
        for i, metadata in enumerate(metadatas):
            metadata["similarity_score"] = float(distances[i])
            metadata["similarity_metric_used"] = similarity_metric
        
        return chunks, metadatas
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return [], []
    
def store_markdown_in_chromadb(
    markdown_content: str,
    chunking_strategy: str = "Semantic Chuking(Kamradt Method)",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    source_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process markdown content, chunk it, and store in ChromaDB.
    
    Args:
        markdown_content: Markdown content to process
        chunking_strategy: Chunking strategy to use
        embedding_model_name: Embedding model to use
        source_info: Source information (file name, path, etc.)
        
    Returns:
        Dictionary with chunking and storage information
    """
    # Default source info if not provided
    if source_info is None:
        source_info = {
            "file_name": "user_markdown",
            "file_path": "user_upload",
            "year": "unknown",
            "quarter": "unknown"
        }
    
    try:
        # Step 1: Chunk the document
        print(f"Chunking document using {chunking_strategy} strategy...")
        chunks = chunk_document(markdown_content, chunking_strategy)
        print(f"Created {len(chunks)} chunks")
        
        # Step 2: Store chunks in ChromaDB
        print(f"Storing chunks in ChromaDB...")
        collection_name, num_stored = store_chunks_in_chromadb(
            chunks, 
            source_info, 
            embedding_model_name
        )
        
        return {
            "status": "success",
            "chunks_total": len(chunks),
            "collection_name": collection_name,
            "chunks_stored": num_stored,
            "source_info": source_info,
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model_name
        }
    
    except Exception as e:
        print(f"Error processing markdown: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "chunks_total": 0,
            "collection_name": None
        }

def query_and_generate_response(
    query: str,
    similarity_metric: str = "cosine",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    llm_model: str = "gpt4o",
    top_k: int = 5,
    data_source: str = None,
    quarters: List[str] = None
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks from ChromaDB and generate a response.
    
    Args:
        query: Query to answer
        similarity_metric: Similarity metric to use
        embedding_model_name: Embedding model to use
        llm_model: LLM model to use for response generation
        top_k: Number of chunks to retrieve
        data_source: Source of data (e.g., "Nvidia Dataset")
        quarters: List of quarters to filter by (e.g., ["2020Q1", "2021Q3"])
        
    Returns:
        Dictionary with response and metadata
    """
    try:
        # Step 1: Determine which collection to use
        collection_name = "nvidia_embeddings" if data_source == "Nvidia Dataset" else COLLECTION_NAME
        print(f"Using collection: {collection_name} for data source: {data_source}")
        
        # Initialize embedding model
        model = SentenceTransformer(EMBEDDING_MODELS[embedding_model_name])
        
        # Generate query embedding
        query_embedding = model.encode(query).tolist()
        
        # Set up ChromaDB client
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        try:
            # Get collection
            collection = client.get_collection(name=collection_name)
            
            # Prepare where clause for filtering if using Nvidia dataset
            where_filter = None
            if data_source == "Nvidia Dataset" and quarters:
                # Parse quarters into year and quarter components
                where_conditions = []
                for quarter_str in quarters:
                    # Handle different possible formats (2020Q1, 2020q1, 2020-Q1)
                    quarter_str = quarter_str.upper().replace('-', '')
                    if 'Q' in quarter_str:
                        year, quarter = quarter_str.split('Q')
                    else:
                        # Format like 2020q1
                        year = quarter_str[:4]
                        quarter = f"Q{quarter_str[5:]}"
                    
                    where_conditions.append({
                        "$and": [
                            {"Year": year},
                            {"Quarter": f"Q{quarter}"}
                        ]
                    })
                
               # FIX: Handle single condition differently
                if len(where_conditions) == 1:
                    # Use the single condition directly without $or
                    where_filter = where_conditions[0]
                    print(f"Applying single filter: {where_filter}")
                elif len(where_conditions) > 1:
                    # Use $or for multiple conditions
                    where_filter = {"$or": where_conditions}
                    print(f"Applying multi-filter: {where_filter}")
            
            # Query collection with filtering by quarters if needed
            if where_filter:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                    where=where_filter
                )
            else:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            
            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Add distance score to metadata
            for i, metadata in enumerate(metadatas):
                metadata["similarity_score"] = float(distances[i])
                metadata["similarity_metric_used"] = similarity_metric
            
            if not chunks:
                return {
                    "response": "No relevant information found for your query in the selected quarters.",
                    "chunks_retrieved": 0,
                    "sources": []
                }
                
            # Step 2: Generate response using LiteLLM
            print(f"Generating response using {llm_model} with {len(chunks)} chunks...")
            
            # Use generate_response function from litellm_query_generator
            llm_response = generate_response(
                chunks=chunks,
                query=query,
                model_id=llm_model,
                metadata=metadatas
            )
            
            # Extract response and usage information
            response_content = llm_response.get("answer", "Error generating response")
            token_usage = llm_response.get("usage", {})
            
            # Format sources for response
            sources = []
            for i, metadata in enumerate(metadatas):
                source_info = {
                    "source": metadata.get("source", "Unknown"),
                    "similarity": metadata.get("similarity_score", 0.0),
                    "document": f"{metadata.get('file_name', 'Unknown')}",
                    "year": metadata.get("year", "Unknown"),
                    "quarter": metadata.get("quarter", "Unknown"),
                    "text": chunks[i][:200] + "..." if len(chunks[i]) > 200 else chunks[i]
                }
                sources.append(source_info)
            
            # Prepare final response
            result = {
                "response": response_content,
                "chunks_retrieved": len(chunks),
                "sources": sources,
                "token_usage": token_usage,
                "collection_used": collection_name
            }
            
            return result
        
        except Exception as e:
            print(f"Error retrieving from collection {collection_name}: {e}")
            return {
                "response": f"Error retrieving information: {str(e)}",
                "chunks_retrieved": 0,
                "sources": []
            }
    
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return {
            "response": f"Error processing your query: {str(e)}",
            "chunks_retrieved": 0,
            "sources": []
        }
    
# For testing
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="RAG Pipeline with configurable components")
    parser.add_argument("--markdown_file", type=str, help="Path to markdown file")
    parser.add_argument("--query", type=str, required=True, help="Query to answer")
    parser.add_argument("--chunking", type=str, default="Semantic Chuking(Kamradt Method)", 
                        choices=["Character-Based Chunking", "Recursive Character/Token Chunking", "Semantic Chuking(Kamradt Method)"],
                        help="Chunking strategy")
    parser.add_argument("--similarity", type=str, default="cosine", 
                        choices=["cosine", "euclidean", "dot_product"],
                        help="Similarity metric")
    parser.add_argument("--embedding", type=str, default="all-MiniLM-L6-v2",
                        choices=list(EMBEDDING_MODELS.keys()),
                        help="Embedding model")
    parser.add_argument("--llm", type=str, default="gpt4o",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="LLM model for response generation")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # Either get markdown from file or stdin
    if args.markdown_file:
        try:
            with open(args.markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Extract file information for source_info
            file_path = os.path.abspath(args.markdown_file)
            file_name = os.path.basename(args.markdown_file)
            
            # Try to extract year and quarter from filename if in format Q1_2020.md
            year = "unknown"
            quarter = "unknown"
            name_parts = file_name.split('_')
            for part in name_parts:
                if part.startswith('Q') and len(part) == 2 and part[1].isdigit():
                    quarter = part
                if part.isdigit() and len(part) == 4:
                    year = part
            
            source_info = {
                "file_name": file_name,
                "file_path": file_path,
                "year": year,
                "quarter": quarter
            }
            
        except Exception as e:
            print(f"Error reading markdown file: {e}")
            sys.exit(1)
    else:
        print("Please provide a markdown file using --markdown_file")
        sys.exit(1)
    
    # Store markdown in ChromaDB
    storage_result = store_markdown_in_chromadb(
        markdown_content,
        args.chunking,
        args.embedding,
        source_info
    )
    
    if storage_result["status"] == "error":
        print(f"Error processing content: {storage_result['error_message']}")
        sys.exit(1)
        
    # Query and generate response
    result = query_and_generate_response(
        args.query,
        args.similarity,
        args.embedding,
        args.llm,
        args.top_k
    )
    
    # Print the results
    print("\n" + "="*80)
    print(f"QUERY: {args.query}")
    print("="*80)
    print(f"RESPONSE:")
    print(result["response"])
    print("\n" + "="*80)
    print(f"METADATA:")
    print(f"  Chunking Strategy: {args.chunking}")
    print(f"  Similarity Metric: {args.similarity}")
    print(f"  Embedding Model: {args.embedding}")
    print(f"  LLM Model: {args.llm}")
    print(f"  Total Chunks: {storage_result['chunks_total']}")
    print(f"  Retrieved Chunks: {result['chunks_retrieved']}")
    print(f"  Collection Used: {result.get('collection_used', 'N/A')}")
    
    if "token_usage" in result:
        print(f"  Token Usage: {result['token_usage']}")
    
    print("\nSOURCES:")
    for i, source in enumerate(result["sources"]):
        print(f"  {i+1}. {source['source']} (Year: {source['year']}, Quarter: {source['quarter']})")
        print(f"     Similarity: {source['similarity']:.4f}")
    
    print("="*80)