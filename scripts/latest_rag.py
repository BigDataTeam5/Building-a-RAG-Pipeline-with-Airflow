import os
import numpy as np
import boto3
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# ✅ Ensure all environment variables are set
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME]):
    raise ValueError("⚠️ Missing required environment variables! Check your .env file.")

# ✅ Initialize AWS S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# ✅ Local storage path
TEXT_DIR = "extracted_texts"
EMBEDDING_PATH = "document_embeddings.pkl"
os.makedirs(TEXT_DIR, exist_ok=True)

# ✅ Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def list_extracted_texts():
    """Lists all extracted text files in the local directory."""
    return [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]

def load_text_chunks():
    """Reads all extracted text files and stores them in a list."""
    texts = []
    filenames = list_extracted_texts()

    for filename in filenames:
        filepath = os.path.join(TEXT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            texts.append((filename, text))
    
    return texts

def generate_embeddings(texts):
    """Generates embeddings for extracted text chunks."""
    text_chunks = [text[1] for text in texts]  # Extract only text part
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    return embeddings

def save_embeddings(embeddings, texts):
    """Saves embeddings and text mapping locally for retrieval."""
    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump((embeddings, texts), f)
    print(f"✅ Saved {len(texts)} document embeddings to {EMBEDDING_PATH}")

def load_embeddings():
    """Loads previously saved embeddings and text mapping."""
    if os.path.exists(EMBEDDING_PATH):
        with open(EMBEDDING_PATH, "rb") as f:
            return pickle.load(f)
    return None, None

def retrieve_relevant_text(query, embeddings, texts, top_k=1):
    """Finds the most relevant text chunk using Cosine Similarity."""
    if embeddings is None or texts is None:
        print("⚠️ No embeddings found! Run the script to generate embeddings first.")
        return None

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]  # Get similarity scores

    # Get the top matching text chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_results = [(texts[i][0], texts[i][1], similarities[i]) for i in top_indices]

    return top_results

if __name__ == "__main__":
    print("\n🔄 Loading extracted text files...")
    extracted_texts = load_text_chunks()

    if extracted_texts:
        print(f"📄 Loaded {len(extracted_texts)} extracted documents.")

        # Generate embeddings only if not already saved
        if not os.path.exists(EMBEDDING_PATH):
            print("\n🔍 Generating embeddings for the extracted texts...")
            document_embeddings = generate_embeddings(extracted_texts)
            save_embeddings(document_embeddings, extracted_texts)
        else:
            print("\n✅ Embeddings already exist. Loading from cache...")

        # Load embeddings and text mappings
        embeddings, texts = load_embeddings()

        # 🔄 **Query & Retrieval**
        while True:
            user_query = input("\n🔎 Enter your query (or type 'exit' to quit): ")
            if user_query.lower() == "exit":
                break

            results = retrieve_relevant_text(user_query, embeddings, texts, top_k=1)
            if results:
                print("\n📖 Most Relevant Document Chunk:")
                print(f"📂 Source: {results[0][0]}")
                print(f"🔍 Similarity Score: {results[0][2]:.4f}")
                print(f"📝 Text:\n{results[0][1]}")
            else:
                print("⚠️ No relevant document found.")
    else:
        print("⚠️ No extracted text files found. Ensure that text extraction is completed first.")
