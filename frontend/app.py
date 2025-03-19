import streamlit as st
import requests

st.title("📄 NVIDIA RAG AI System")

# Select the PDF Parser
parser_choice = st.selectbox("Select PDF Parser", ["Docling", "Mistral OCR"])

# Select RAG method
rag_method = st.radio("Choose RAG method", ["Naive", "Pinecone", "ChromaDB"])

# Select Chunking Strategy
chunking_strategy = st.radio("Select Chunking Strategy", ["Fixed-size", "Semantic", "Overlapping"])

# Query Input
query = st.text_input("🔍 Enter your query")

# Fetch results
if st.button("Retrieve"):
    url = f"http://localhost:8000/retrieve/?query={query}&method={rag_method}"
    response = requests.get(url).json()
    st.write("🔎 Results:")
    st.json(response)
