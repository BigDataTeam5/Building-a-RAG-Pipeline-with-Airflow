import chromadb
from chromadb.utils import embedding_functions
import sqlite3
import os
import json

def inspect_chromadb():
    """Inspect ChromaDB files and collection details"""
    
    # Define paths to check - prioritize local paths for development
    paths_to_check = [
        "./chroma_db",                 # Local relative path
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db"),  # Project root path
        "/opt/airflow/chroma_db",      # Airflow container path (fallback)
        "/app/chroma_db",              # FastAPI container path (fallback)
    ]
    
    print("\n===== CHECKING CHROMADB FILES =====")
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"\n✅ Found ChromaDB at: {path}")
            print(f"Contents: {os.listdir(path)}")
            
            # Check SQLite database directly
            db_path = os.path.join(path, "chroma.sqlite3")
            if os.path.exists(db_path):
                try:
                    print(f"\nInspecting SQLite database at: {db_path}")
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # List all tables
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    print("\nDatabase tables:")
                    for table in tables:
                        print(f"- {table[0]}")
                        
                        # Show schema for each table
                        cursor.execute(f"PRAGMA table_info({table[0]})")
                        columns = cursor.fetchall()
                        print("  Columns:")
                        for col in columns:
                            print(f"    {col[1]} ({col[2]})")
                        
                        # Show row count for each table
                        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                        count = cursor.fetchone()[0]
                        print(f"  Row count: {count}")
                        
                        # For collections table, show detailed info
                        if table[0] == 'collections':
                            print("\n  Collections Table Details:")
                            cursor.execute("SELECT * FROM collections")
                            all_collections = cursor.fetchall()
                            for collection in all_collections:
                                print(f"    Collection: {collection}")
                    
                    conn.close()
                except Exception as e:
                    print(f"Error inspecting SQLite database: {str(e)}")
            
            # Try to access via ChromaDB API
            try:
                print(f"\nTrying ChromaDB API access for: {path}")
                client = chromadb.PersistentClient(path=path)
                collections = client.list_collections()
                print(f"\nFound {len(collections)} collections:")
                
                for collection in collections:
                    print(f"\n----- Collection: {collection.name} -----")
                    try:
                        # Try with embedding function
                        ef = embedding_functions.DefaultEmbeddingFunction()
                        coll = client.get_collection(
                            name=collection.name,
                            embedding_function=ef
                        )
                        count = coll.count()
                        print(f"Collection has {count} documents")
                        
                        if count > 0:
                            # Get sample documents
                            result = coll.get(limit=1)
                            print("\nSample Document:")
                            print(f"Metadata: {json.dumps(result['metadatas'][0], indent=2)}")
                            print(f"Document Preview: {result['documents'][0][:200]}...")
                            
                            # Try a test query
                            query = "What was NVIDIA's revenue?"
                            print(f"\nTesting query: {query}")
                            query_results = coll.query(
                                query_texts=[query],
                                n_results=2,
                                include=["documents", "metadatas", "distances"]
                            )
                            if query_results['documents'][0]:
                                print("\nQuery Results:")
                                print(f"Found {len(query_results['documents'][0])} matches")
                                print(f"First match: {query_results['documents'][0][0][:200]}...")
                    
                    except Exception as e:
                        print(f"Error accessing collection: {str(e)}")
                
            except Exception as e:
                print(f"Error with ChromaDB API: {str(e)}")
        else:
            print(f"❌ Path not found: {path}")

if __name__ == "__main__":
    print("Starting ChromaDB Inspection...")
    inspect_chromadb()