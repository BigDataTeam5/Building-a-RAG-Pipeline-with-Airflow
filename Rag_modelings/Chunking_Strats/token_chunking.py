# Install the chunking_evaluation library
#!pip install git+https://github.com/brandonstarxel/chunking_evaluation.git

# Import the necessary chunking functions
from chunking_evaluation.chunking import (
    FixedTokenChunker,
    RecursiveTokenChunker,
    KamradtModifiedChunker,
    ClusterSemanticChunker,
    LLMSemanticChunker
)

# Additional dependencies
import os
import json
import tiktoken
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions
from chunking_evaluation.utils import openai_token_count
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom visualization helper
from chunk_visualizer import (
    visualize_chunks_html, 
    analyze_chunks_stats, 
    plot_chunk_stats, 
    save_chunks_to_json,
    setup_chunking_output
)

# Create the base output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load the sample document
file_path = "../../Q1.md"
with open(file_path, 'r', encoding='utf-8') as file:
    document = file.read()

# Print the first 500 characters to see what we're working with
print("First 500 characters: ", document[:500])

# Get the total token count
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(document)
print(f"Total document length: {len(tokens)} tokens")

def analyze_chunks(chunks, use_tokens=False):
    """
    Analyze a list of chunks to show statistics and overlaps.
    
    Args:
        chunks: List of text chunks
        use_tokens: Whether to analyze overlap by tokens instead of characters
    """
    # Print basic stats
    print("\nNumber of Chunks:", len(chunks))
    
    # Show examples of chunks
    if len(chunks) >= 2:
        print("\n", "="*50, f"Chunk #{len(chunks)//3}", "="*50)
        print(chunks[len(chunks)//3])
        print("\n", "="*50, f"Chunk #{2*len(chunks)//3}", "="*50)
        print(chunks[2*len(chunks)//3])
    
    # Calculate average chunk size
    if use_tokens:
        encoding = tiktoken.get_encoding("cl100k_base")
        chunk_sizes = [len(encoding.encode(chunk)) for chunk in chunks]
        print(f"\nAverage chunk size: {sum(chunk_sizes)/len(chunk_sizes):.1f} tokens")
        print(f"Min chunk size: {min(chunk_sizes)} tokens")
        print(f"Max chunk size: {max(chunk_sizes)} tokens")
    else:
        chunk_sizes = [len(chunk) for chunk in chunks]
        print(f"\nAverage chunk size: {sum(chunk_sizes)/len(chunk_sizes):.1f} characters")
        print(f"Min chunk size: {min(chunk_sizes)} characters")
        print(f"Max chunk size: {max(chunk_sizes)} characters")
    
    # Find overlaps if there are at least two chunks
    if len(chunks) >= 2:
        chunk1, chunk2 = chunks[len(chunks)//2], chunks[len(chunks)//2 + 1]
        
        if use_tokens:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens1 = encoding.encode(chunk1)
            tokens2 = encoding.encode(chunk2)
            
            # Find overlapping tokens
            for i in range(min(len(tokens1), 50), 0, -1):
                if tokens1[-i:] == tokens2[:i]:
                    overlap = encoding.decode(tokens1[-i:])
                    print("\n", "="*50, f"\nOverlapping text ({i} tokens):", overlap)
                    return
            print("\nNo token overlap found")
        else:
            # Find overlapping characters
            for i in range(min(len(chunk1), 200), 0, -1):
                if chunk1[-i:] == chunk2[:i]:
                    print("\n", "="*50, f"\nOverlapping text ({i} chars):", chunk1[-i:])
                    return
            print("\nNo character overlap found")

def character_chunk_text(document, chunk_size, overlap):
    """
    Split text into chunks based on character count.
    
    Args:
        document: Text to split
        chunk_size: Maximum number of characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    stride = chunk_size - overlap
    current_idx = 0
    
    while current_idx < len(document):
        # Take chunk_size characters starting from current_idx
        chunk = document[current_idx:current_idx + chunk_size]
        if not chunk:  # Break if we're out of text
            break
        chunks.append(chunk)
        current_idx += stride  # Move forward by stride
    
    return chunks

def save_chunks_to_json(chunks, strategy_name, file_path):
    """
    Save chunks to a JSON file with the strategy name and file path.
    
    Args:
        chunks: List of text chunks
        strategy_name: Name of the chunking strategy
        file_path: Path of the file being chunked
    """
    output_dir = setup_chunking_output(strategy_name)
    output_file = os.path.join(output_dir, "chunks.json")
    
    data = {
        "strategy": strategy_name,
        "chunk_count": len(chunks),
        "chunks": [{"id": i, "file_path": file_path,"text": chunk, "char_length": len(chunk), "token_length": openai_token_count(chunk)} for i, chunk in enumerate(chunks)]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Chunks saved to {output_file}")


# Set up output directory for token chunking
strategy_name = "token_chunking"
output_dir = setup_chunking_output(strategy_name)

# Using FixedTokenChunker from the chunking_evaluation library
fixed_token_chunker = FixedTokenChunker(
    chunk_size=400,  # 400 tokens per chunk 
    chunk_overlap=0,  # No overlap
    encoding_name="cl100k_base"  # Use OpenAI's cl100k tokenizer
)

token_chunks = fixed_token_chunker.split_text(document)
analyze_chunks(token_chunks, use_tokens=True)

# Save chunks to JSON
save_chunks_to_json(token_chunks, strategy_name, file_path)

# With overlap
token_overlap_chunker = FixedTokenChunker(
    chunk_size=400, 
    chunk_overlap=200,  # 200 token overlap
    encoding_name="cl100k_base"
)

token_overlap_chunks = token_overlap_chunker.split_text(document)
analyze_chunks(token_overlap_chunks, use_tokens=True)

# Save overlap chunks to JSON
save_chunks_to_json(token_overlap_chunks, f"{strategy_name}_with_overlap", file_path)

# Visualize token chunks
visualize_chunks_html(
    document, 
    token_chunks, 
    output_path="visualization.html", 
    title="Token-Based Chunking", 
    strategy_name=strategy_name
)

# Generate and plot statistics
stats = analyze_chunks_stats(token_chunks, use_tokens=True)
plot_chunk_stats(
    stats, 
    title="Token-Based Chunking Stats", 
    output_path="stats.png", 
    strategy_name=strategy_name
)
