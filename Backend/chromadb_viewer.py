import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

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
    
    # Use first two documents for comparison
    doc_ids = result['ids'][:2]  # Get first two documents
    if len(doc_ids) < 2:
        print("Not enough documents in collection for comparison")
        exit(1)
        
    doc_id_1, doc_id_2 = doc_ids
    print(f"\nComparing documents:")
    print(f"Document 1 ID: {doc_id_1}")
    print(f"Document 2 ID: {doc_id_2}")
    
    similarity_result = calculate_similarity(collection, doc_id_1, doc_id_2)
    if similarity_result:
        similarity_score, doc1_data, doc2_data = similarity_result
        print("\nDocument 1:")
        print(f"Preview: {doc1_data['document'][:100]}...")
        print(f"Metadata: {doc1_data['metadata']}")
        
        print("\nDocument 2:")
        print(f"Preview: {doc2_data['document'][:100]}...")
        print(f"Metadata: {doc2_data['metadata']}")
        
        print(f"\nCosine similarity between documents: {similarity_score:.4f}")
    else:
        print("\nError calculating similarity between documents")