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
with open("177440d5-3b32-4185-8cc8-95500a9dc783-with-images.md", 'r', encoding='utf-8') as file:
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


# Check if OpenAI API key is set
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable is not set. Skipping semantic chunking strategies.")
else:
    # Set up output directory for semantic chunking
    strategy_name = "kamradt_chunking"
    output_dir = setup_chunking_output(strategy_name)
    
    # Set up an embedding function
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, 
        model_name="text-embedding-3-small"
    )
    
    # Using KamradtModifiedChunker from the chunking_evaluation library
    kamradt_chunker = KamradtModifiedChunker(
        avg_chunk_size=300,      # Target size in tokens
        min_chunk_size=50,       # Initial split size
        embedding_function=embedding_function  # Pass your embedding function
    )
    
    kamradt_chunks = kamradt_chunker.split_text(document)
    analyze_chunks(kamradt_chunks, use_tokens=True)
    
    # Save chunks to JSON
    save_chunks_to_json(kamradt_chunks, strategy_name)
    
    # Visualize semantic chunks
    visualize_chunks_html(
        document, 
        kamradt_chunks, 
        output_path="visualization.html", 
        title="Kamradt Chunking", 
        strategy_name=strategy_name
    )
    

