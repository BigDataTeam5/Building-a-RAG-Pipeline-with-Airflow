import os
import chromadb
import sqlite3
import shutil
import time
from chromadb.utils import embedding_functions

def fix_chromadb_schema():
    """Fix ChromaDB schema without losing data"""
    CHROMA_DB_PATH = "/app/chroma_db"
    DB_FILE = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    
    print(f"Attempting to fix ChromaDB at {CHROMA_DB_PATH}")
    
    if not os.path.exists(DB_FILE):
        print(f"Database file not found at {DB_FILE}")
        print(f"Looking for ChromaDB files in {CHROMA_DB_PATH}...")
        if os.path.exists(CHROMA_DB_PATH):
            files = os.listdir(CHROMA_DB_PATH)
            print(f"Found files: {files}")
        return False
    
    # Create backup
    backup_file = f"{DB_FILE}.backup_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(DB_FILE, backup_file)
        print(f"Created backup at {backup_file}")
    except Exception as e:
        print(f"Could not create backup: {e}")
    
    # Try to directly fix the schema with SQL
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if the topic column exists
        cursor.execute("PRAGMA table_info(collections)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Collections table columns: {columns}")
        
        if 'topic' not in columns:
            print("Adding missing 'topic' column to collections table")
            cursor.execute("ALTER TABLE collections ADD COLUMN topic TEXT")
            conn.commit()
            print("Schema fixed!")
        else:
            print("Collections table already has topic column")
        
        conn.close()
        print("Database schema updated successfully")
        return True
    except Exception as e:
        print(f"Error fixing schema: {e}")
        return False

if __name__ == "__main__":
    fix_chromadb_schema()