import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import sys
import shutil

# NumPy 2.0+ compatibility for ChromaDB
if hasattr(np, '__version__') and np.__version__.startswith('2.'):
    np.float_ = np.float64
    np.int_ = np.int64
    np.uint = np.uint64
    print("Applied NumPy 2.0 compatibility fixes for ChromaDB")

# Add the project root to sys.path to import from other directories
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import the response generator
try:
    from Backend.litellm_query_generator import generate_response
except ImportError:
    print("Warning: litellm_query_generator not found, response generation will be disabled")
    generate_response = None

# Use the Docker copied database path instead of the local path
CHROMA_DB_PATH = os.path.join(project_root, "docker_chroma_db")
if not os.path.exists(CHROMA_DB_PATH):
    # Fall back to the default path if the Docker copy doesn't exist
    CHROMA_DB_PATH = os.path.join(project_root, "chroma_db")

print(f"Using ChromaDB path: {CHROMA_DB_PATH}")

# Initialize client with proper settings
try:
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=chromadb.Settings(anonymized_telemetry=False)
    )
except Exception as e:
    print(f"Error initializing ChromaDB client: {str(e)}")
    client = None

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_available_collections():
    if not client:
        return None
        
    try:
        collection_names = client.list_collections()
        if not collection_names:
            print("No collections found in the database.")
            return None
        print(f"Available collections: {[c.name for c in collection_names]}")
        return collection_names
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
        return None

def get_collection_info(collection_name="nvidia_embeddings"):
    if not client:
        return None, None
        
    try:
        collection = client.get_collection(name=collection_name)
        # Get all documents to verify content
        result = collection.get(limit=100)  # Get a reasonable sample
        print(f"\nCollection '{collection_name}' stats:")
        print(f"Total documents: {len(result['ids'])}")
        print(f"First few document IDs: {result['ids'][:5]}")
        print("\nMetadata preview:")
        for i in range(min(3, len(result['metadatas']))):
            print(f"Document {i+1}: {result['metadatas'][i]}")
        return collection, result
    except Exception as e:
        print(f"Error accessing collection: {str(e)}")
        return None, None

def test_nvidia_query(query="what is nvidia", quarters=["2022Q1", "2022Q4"]):
    """
    Test a query against the nvidia_embeddings collection with quarter filtering.
    
    Args:
        query: The query to test
        quarters: List of quarters to filter by
    """
    if not client:
        return
        
    try:
        # Initialize model and client
        model = SentenceTransformer(EMBEDDING_MODEL)
        collection = client.get_collection(name="nvidia_embeddings")
        
        # Generate query embedding
        query_embedding = model.encode(query).tolist()
        
        # First, test without filters to see what's available
        print("\n--- Testing unfiltered query ---")
        unfiltered_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"Unfiltered query returned {len(unfiltered_results['documents'][0])} results")
        if len(unfiltered_results['documents'][0]) > 0:
            print("Sample metadata without filtering:")
            print(unfiltered_results['metadatas'][0][0])
        
        # Build where conditions for the quarters
        where_conditions = []
        for quarter_str in quarters:
            # Normalize quarter format
            quarter_str = quarter_str.upper().replace('-', '')
            if 'Q' in quarter_str:
                year, quarter = quarter_str.split('Q')
            else:
                year = quarter_str[:4]
                quarter = quarter_str[4:]
            
            # IMPORTANT: Check actual metadata structure in first test
            metadata_sample = unfiltered_results['metadatas'][0][0] if unfiltered_results['metadatas'][0] else {}
            print(f"\nGot metadata sample: {metadata_sample}")
            
            # Use the correct field names from actual metadata
            year_field = "Year" if "Year" in metadata_sample else "year"
            quarter_field = "Quarter" if "Quarter" in metadata_sample else "quarter"
            
            print(f"Using fields: {year_field}={year}, {quarter_field}=Q{quarter}")
            
            where_conditions.append({
                "$and": [
                    {year_field: year},
                    {quarter_field: f"Q{quarter}"}
                ]
            })
        
        # Create the where filter
        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
            print(f"\nApplying single filter: {where_filter}")
        elif len(where_conditions) > 1:
            where_filter = {"$or": where_conditions}
            print(f"\nApplying multi-filter: {where_filter}")
        
        # Query with filter
        print("\n--- Testing filtered query ---")
        filtered_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
            where=where_filter
        )
        
        print(f"Filtered query returned {len(filtered_results['documents'][0])} results")
        
        if len(filtered_results['documents'][0]) > 0:
            # Use LiteLLM to generate a response
            chunks = filtered_results["documents"][0]
            metadatas = filtered_results["metadatas"][0]
            
            print("\n--- Sample chunks retrieved ---")
            for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                print(f"\nChunk {i+1} Preview:")
                print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                print(f"Metadata: {metadatas[i]}")
            
            if generate_response:
                print("\n--- Generating response ---")
                response = generate_response(
                    chunks=chunks,
                    query=query,
                    model_id="gpt4o",
                    metadata=metadatas
                )
                
                print("\nFinal Answer:")
                print(response.get("answer", "No answer generated"))
            else:
                print("\nResponse generation disabled (litellm_query_generator not found)")
        else:
            print("\nNo matching documents found with the specified quarters.")
            
    except Exception as e:
        print(f"Error in test query: {str(e)}")
        import traceback
        traceback.print_exc()

def show_documents_in_collection(collection_name="nvidia_embeddings", limit=20):
    """Display documents from the collection"""
    if not client:
        return
        
    try:
        collection = client.get_collection(name=collection_name)
        result = collection.get(limit=limit)
        
        print(f"\n===== DISPLAYING {min(limit, len(result['ids']))} DOCUMENTS FROM {collection_name} =====")
        
        for i in range(min(limit, len(result['ids']))):
            print(f"\n----- Document {i+1} -----")
            print(f"ID: {result['ids'][i]}")
            if result['metadatas'] and result['metadatas'][i]:
                metadata = result['metadatas'][i]
                print(f"Year: {metadata.get('Year', metadata.get('year', 'Unknown'))}")
                print(f"Quarter: {metadata.get('Quarter', metadata.get('quarter', 'Unknown'))}")
                print(f"Source: {metadata.get('source', 'Unknown')}")
            
            if result['documents'] and i < len(result['documents']):
                doc = result['documents'][i]
                preview = doc[:150] + "..." if len(doc) > 150 else doc
                print(f"Text: {preview}")
    
    except Exception as e:
        print(f"Error displaying documents: {str(e)}")

if __name__ == "__main__":
    print(f"Using ChromaDB path: {CHROMA_DB_PATH}")
    
    # First, verify collections
    collection_names = get_available_collections()
    if not collection_names:
        print("No collections available. Please ensure the database is properly initialized.")
        exit(1)
    
    # Get collection and its contents
    try:
        collection_name = collection_names[0].name
        print(f"Using collection: {collection_name}")
        collection, result = get_collection_info(collection_name)
        if not collection or not result:
            print("Failed to access collection.")
            exit(1)
        
        # Display some documents from the collection
        show_documents_in_collection(collection_name, limit=10)
        
        # Ask if user wants to test a query
        user_input = input("\nDo you want to test a query? (y/n): ")
        if user_input.lower() == 'y':
            query = input("Enter your query: ") or "what is nvidia revenue for year 2021 quarter 1"
            quarters = input("Enter quarters to search (comma separated, e.g. 2021Q1,2021Q2): ") or "2021Q1"
            quarters_list = [q.strip() for q in quarters.split(",")]
            test_nvidia_query(query=query, quarters=quarters_list)
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()