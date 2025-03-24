Understanding the RAG Pipeline Flow with Pinecone Namespaces
Current Pipeline Flow
Your RAG pipeline follows these key steps:

Document Ingestion & Preprocessing

Load markdown content
Extract document title, headings, section structure
Process document with selected chunking strategy (semantic, recursive, etc.)
Chunking & Storage

Chunks are created using your selected method (Kamradt, recursive, character-based)
Full chunk content is saved to a JSON file for later retrieval
Each chunk gets a unique ID (hash-based) for reference
Embedding & Indexing

Generate embeddings using selected model (text-embedding-ada-002)
Create rich metadata (preview, keywords, section info, etc.)
Upload vectors to Pinecone in batches
Query Analysis

Analyze query intent (TOC query, specific document, etc.)
Extract keywords for hybrid search
Create smart filters based on intent
Hybrid Search & Retrieval

Search Pinecone for semantically similar vectors
Apply hybrid reranking (combining vector similarity + keyword matching)
Boost scores for document title matches, TOC relevance, etc.
Retrieve full chunk content from JSON storage
Response Generation

Enhance query with detected context
Send to LiteLLM for response generation
Format response with source information