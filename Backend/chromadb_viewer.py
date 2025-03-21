import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import sys

# Add the project root to sys.path to import from other directories
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import the response generator
from Backend.litellm_query_generator import generate_response

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_available_collections():
    try:
        # Updated for ChromaDB v0.6.0
        collection_names = client.list_collections()
        if not collection_names:
            print("No collections found in the database.")
            return None
        print(f"Available collections: {collection_names}")
        return collection_names
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
        return None

def get_collection_info(collection_name="nvidia_embeddings"):
    try:
        collection = client.get_collection(name=collection_name)
        # Get all documents to verify content
        result = collection.get()
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
            print("\nNo matching documents found with the specified quarters.")
            
    except Exception as e:
        print(f"Error in test query: {str(e)}")

def calculate_similarity(collection, doc_id_1, doc_id_2):
    """Calculate similarity between two documents in the collection"""
    doc1_data = get_embedding_for_document(collection, doc_id_1)
    doc2_data = get_embedding_for_document(collection, doc_id_2)
    
    if doc1_data and doc2_data:
        # Calculate cosine similarity
        embedding1 = np.array(doc1_data['embedding']).reshape(1, -1)
        embedding2 = np.array(doc2_data['embedding']).reshape(1, -1)
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity, doc1_data, doc2_data
    return None

def get_embedding_for_document(collection, doc_id):
    try:
        # Include all data types in the query
        result = collection.get(
            ids=[doc_id],
            include=['embeddings', 'documents', 'metadatas']
        )
        if result and len(result['embeddings']) > 0:
            return {
                'embedding': result['embeddings'][0],
                'document': result['documents'][0],
                'metadata': result['metadatas'][0] if result['metadatas'] else None
            }
        print(f"No data found for document ID: {doc_id}")
        return None
    except Exception as e:
        print(f"Error retrieving data for {doc_id}: {str(e)}")
        return None

if __name__ == "__main__":
    print(f"Using ChromaDB path: {CHROMA_DB_PATH}")
    
    # First, verify collections
    collection_names = get_available_collections()
    if not collection_names:
        print("No collections available. Please ensure the database is properly initialized.")
        exit(1)
    
    # Get collection and its contents
    collection, result = get_collection_info("nvidia_embeddings")
    if not collection or not result:
        print("Failed to access nvidia_embeddings collection.")
        exit(1)
    
    # Test query with specific quarters
    test_nvidia_query(query="what is nvidia", quarters=["2022Q1", "2022Q4"])