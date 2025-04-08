import os
import chromadb
from chromadb.utils import embedding_functions
import shutil
import time

def fix_chromadb_collection():
    """Fix ChromaDB collection schema issues"""
    # Path to ChromaDB directory
    CHROMA_DB_PATH = "/app/chroma_db"
    print(f"Fixing ChromaDB at {CHROMA_DB_PATH}")
    
    # Create backup of current DB
    backup_path = "/app/chroma_db_backup_" + time.strftime("%Y%m%d_%H%M%S")
    if os.path.exists(CHROMA_DB_PATH):
        try:
            print(f"Creating backup at {backup_path}")
            shutil.copytree(CHROMA_DB_PATH, backup_path)
            print("Backup created successfully")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")
    
    try:
        # Initialize client
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Try to create a new collection with correct schema
        try:
            print("Creating new nvidia_embeddings collection...")
            # Use same embedding function as in Airflow DAG
            ef = embedding_functions.DefaultEmbeddingFunction()
            
            # Delete if exists
            try:
                existing_collections = client.list_collections()
                for collection in existing_collections:
                    if collection.name == "nvidia_embeddings":
                        print(f"Deleting existing collection: {collection.name}")
                        client.delete_collection("nvidia_embeddings")
                        print("Collection deleted")
            except Exception as e:
                print(f"No collection to delete: {e}")
            
            # Create fresh collection
            collection = client.create_collection(
                name="nvidia_embeddings",
                embedding_function=ef,
                metadata={"description": "NVIDIA markdown documents with embeddings"}
            )
            print("New collection created successfully")
            
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        return False

if __name__ == "__main__":
    fix_chromadb_collection()