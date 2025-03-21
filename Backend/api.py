from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional, Any
import os
import uuid
import asyncio
from logger import api_logger, pdf_logger, error_logger, log_request, log_error
import uvicorn
import sys

# Fix path handling for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)  # Add project root to path

# Local imports
from litellm_query_generator import generate_response, MODEL_CONFIGS
from parsing_methods.doclingparsing import main as docling_parse
from parsing_methods.mistralparsing import process_pdf as mistral_parse

# Import modules with absolute paths
from Rag_modelings.chromadb_pipeline import (
    store_markdown_in_chromadb,
    query_and_generate_response as chromadb_query
)
from Rag_modelings.rag_pinecone import (
    query_pinecone_rag,  # Changed from run_rag_pipeline to query_pinecone_rag
    initialize_connections,
    prepare_vectors_for_upload,
    upload_vectors_to_pinecone,
    process_document_with_chunking,
)


# Initialize FastAPI app
app = FastAPI(title="RAG Pipeline API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
UPLOAD_DIR = "uploads"
MARKDOWN_DIR = "user_markdowns"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

# In-memory storage
job_store = {}
embedding_store = {}

# Pydantic models
class DateRequest(BaseModel):
    date: str  # Date format YYYY-MM-DD

class RAGRequest(BaseModel):
    request_id: str
    question: str
    model: str
    parser: Optional[str] = None
    rag_method: Optional[str] = None
    chunking_strategy: Optional[str] = None
    quarters: Optional[List[str]] = None

class EmbeddingRequest(BaseModel):
    markdown_path: str
    rag_method: str  # "chromadb" or "pinecone"
    chunking_strategy: Optional[str] = "Semantic Chuking(Kamradt Method)"
    embedding_model: Optional[str] = "all-MiniLM-L6-v2"

class ManualEmbeddingRequest(BaseModel):
    text: str
    embedding_id: str
    rag_method: str  # "chromadb" or "pinecone"
    chunking_strategy: Optional[str] = "Semantic Chuking(Kamradt Method)"
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    embedding_id: Optional[str] = None  # For manual embeddings
    markdown_path: Optional[str] = None  # For stored markdown files
    rag_method: str  # "chromadb" or "pinecone"
    data_source: Optional[str] = None  # "Nvidia Dataset" or "PDF Upload"
    quarters: Optional[List[str]] = None  # List of quarters to filter by
    model_id: str = "gpt4o"
    similarity_metric: Optional[str] = "cosine"
    top_k: Optional[int] = 5
    
# Function to determine the quarter
def get_quarter(date: str) -> str:
    dt = datetime.strptime(date, "%Y-%m-%d")
    year = dt.year
    quarter = (dt.month - 1) // 3 + 1
    return f"{year}q{quarter}"

@app.get("/")
def read_root():
    return {"message": "RAG Pipeline API is running"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available"}

@app.post("/get_quarter")
def get_year_quarter(date_request: DateRequest):
    try:
        quarter_str = get_quarter(date_request.date)
        return {"year_quarter": quarter_str}
    except Exception as e:
        log_error(f"Error in get_quarter endpoint", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    parser: str = Query("docling"),
    rag_method: Optional[str] = Query(None),
    chunking_strategy: Optional[str] = Query(None)
):
    try:
        file_id = str(uuid.uuid4())
        filename = file.filename
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")

        # Save the uploaded PDF
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Generate the markdown path
        markdown_filename = os.path.splitext(filename)[0] + ".md"
        markdown_path = os.path.join(MARKDOWN_DIR, f"{file_id}_{markdown_filename}")

        # Parse the PDF
        parsed_content = ""
        if parser.lower() == "docling":
            parsed_content = docling_parse(file_path)
        elif parser.lower() == "mistral":
            parsed_content = mistral_parse(file_path)

        # <-- CHANGED: Always write the Markdown file, even if empty
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(parsed_content or "")

        return {
            "filename": filename,
            "saved_as": file_path,
            "file_id": file_id,
            "markdown_path": markdown_path,  # Return this so the front-end can embed
            "parser": parser,
            "rag_method": rag_method,
            "chunking_strategy": chunking_strategy,
            "status": "success"
        }

    except Exception as e:
        log_error(f"Error uploading PDF", e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/nvidia/quarters")
async def get_nvidia_quarters():
    try:
        # Get available NVIDIA quarterly reports (last 5 years)
        current_year = datetime.now().year
        quarters = []
        for year in range(current_year-4, current_year+1):
            for q in range(1, 5):
                if year == current_year and q > ((datetime.now().month - 1) // 3 + 1):
                    continue
                quarters.append(f"{year}q{q}")
        
        api_logger.info(f"Returned {len(quarters)} available quarters")
        return {"quarters": quarters}
    except Exception as e:
        log_error("Error fetching Nvidia quarters", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm/models")
async def get_available_models():
    # Return a list of supported LLM models
    models = list(MODEL_CONFIGS.keys())
    api_logger.info(f"Available LLM models: {', '.join(models)}")
    return {"models": models}

@app.post("/rag/query")
async def rag_query(request: QueryRequest):
    """Query the RAG system using the specified method and model"""
    try:
        log_request(f"RAG Query: {request.query}, Method: {request.rag_method}, Model: {request.model_id},DataSource: {request.data_source}")
        
        result = None
        
        # Handle manual embeddings
        if request.rag_method.lower() == "manual_embedding":
            # For manual embedding method, we'll use LiteLLM to generate a response
            # without retrieval since there may not be specific content to reference
            response = generate_response(
                chunks=["This is a direct query without retrieval."],  # Placeholder
                query=request.query,
                model_id=request.model_id,
                metadata=[{"source": "direct_query"}]
            )
            
            result = {
                "answer": response.get("answer", "Error generating response"),
                "usage": response.get("usage", {}),
                "source": "Manual embedding (direct query)",
                "chunks_used": 1
            }
        
        # Handle manual embeddings
        elif request.embedding_id and request.embedding_id in embedding_store:
            embedding_data = embedding_store[request.embedding_id] 
            chunks = embedding_data["chunks"]
            
            # Use LiteLLM to generate response directly
            response = generate_response(
                chunks=chunks,
                query=request.query,
                model_id=request.model_id,
                metadata=[embedding_data["metadata"] for _ in chunks]
            )
            
            result = {
                "answer": response.get("answer", "Error generating response"),
                "usage": response.get("usage", {}),
                "source": f"Manual embedding (ID: {request.embedding_id})",
                "chunks_used": len(chunks)
            }
        
        # Handle ChromaDB
        elif request.rag_method.lower() == "chromadb":
            result = chromadb_query(
                query=request.query,
                similarity_metric=request.similarity_metric,
                llm_model=request.model_id,
                top_k=request.top_k,
                data_source=request.data_source,
                quarters=request.quarters           
            )   
        
        # Handle Pinecone
        elif request.rag_method.lower() == "pinecone":
            # Use query_pinecone_rag instead of run_rag_pipeline
            response = query_pinecone_rag(
                query=request.query,
                model_id=request.model_id,
                similarity_metric=request.similarity_metric  # Pass similarity_metric from request
            )
            
            # Format the result to match the expected structure
            result = {
                "answer": response.get("answer", "Error generating response"),
                "usage": response.get("usage", {}),
                "sources": response.get("sources", []),
                "similarity_metric_used": response.get("similarity_metric_used", "cosine")
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported RAG method: {request.rag_method}")
        
        return result
    except Exception as e:
        log_error(f"Error processing RAG query", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-llm-result/{request_id}")
async def get_llm_result(request_id: str):
    try:
        if request_id not in job_store:
            raise HTTPException(status_code=404, detail="Request not found")
        
        return job_store[request_id]
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        log_error(f"Error getting LLM result", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/config")
async def set_rag_config(parser: str, rag_method: str, chunking_strategy: str):
    try:
        # This would save the configuration, perhaps trigger Airflow DAG setup
        return {
            "status": "success",
            "config": {
                "parser": parser,
                "rag_method": rag_method,
                "chunking_strategy": chunking_strategy
            }
        }
    except Exception as e:
        log_error(f"Error setting RAG config", e)
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint for creating embeddings from markdown
@app.post("/rag/create-embeddings")
async def create_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """Create embeddings from a markdown file using ChromaDB or Pinecone"""
    try:
        log_request(f"Creating embeddings for {request.markdown_path} using {request.rag_method}")
        
        # Verify the markdown file exists
        markdown_path = request.markdown_path
        if not os.path.exists(markdown_path):
            markdown_path = os.path.join(MARKDOWN_DIR, os.path.basename(request.markdown_path))
            if not os.path.exists(markdown_path):
                raise HTTPException(status_code=404, detail=f"Markdown file not found: {request.markdown_path}")
        
        # Create a job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        job_store[job_id] = {
            "status": "processing",
            "markdown_path": markdown_path,
            "rag_method": request.rag_method,
            "chunking_strategy": request.chunking_strategy
        }
        
        # Process embeddings in the background
        background_tasks.add_task(
            process_embeddings,
            job_id,
            markdown_path,
            request.rag_method,
            request.chunking_strategy,
            request.embedding_model
        )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"Creating embeddings for {os.path.basename(markdown_path)} using {request.rag_method}"
        }
    except Exception as e:
        log_error(f"Error creating embeddings", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/manual-embedding")
async def create_manual_embedding(request: ManualEmbeddingRequest):
    """Create embeddings from text provided directly by the user"""
    try:
        log_request(f"Creating manual embedding with ID {request.embedding_id}")
        
        # Set default metadata if not provided
        metadata = request.metadata or {
            "source": "manual_input",
            "timestamp": datetime.now().isoformat()
        }
        
        # Apply chunking
        chunks = process_document_with_chunking(request.text, request.chunking_strategy)
        
        # Store in embedding_store for later use
        embedding_store[request.embedding_id] = {
            "chunks": chunks,
            "rag_method": request.rag_method,
            "chunking_strategy": request.chunking_strategy,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "embedding_id": request.embedding_id,
            "chunks_count": len(chunks),
            "rag_method": request.rag_method,
            "status": "completed"
        }
    except Exception as e:
        log_error(f"Error creating manual embedding", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an embedding job"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_store[job_id]

@app.get("/rag/embeddings")
async def list_embeddings():
    """List all manual embeddings stored in memory"""
    return {
        "embeddings": [
            {
                "id": k,
                "chunks_count": len(v["chunks"]),
                "rag_method": v["rag_method"],
                "timestamp": v["timestamp"]
            } 
            for k, v in embedding_store.items()
        ]
    }

# Background task for processing embeddings
async def process_embeddings(
    job_id: str,
    markdown_path: str,
    rag_method: str,
    chunking_strategy: str,
    embedding_model: str = None 
):
    try:
        # Read markdown content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Extract file information
        file_name = os.path.basename(markdown_path)
        source_info = {
            "file_name": file_name,
            "file_path": markdown_path,
        }
        
        # Process with ChromaDB
        if rag_method.lower() == "chromadb":
            result = store_markdown_in_chromadb(
                markdown_content,
                chunking_strategy,
                embedding_model=embedding_model,
                source_info=source_info
            )
            
            # Update job status
            job_store[job_id].update({
                "status": "completed" if result["status"] == "success" else "failed",
                "chunks_total": result.get("chunks_total", 0),
                "collection_name": result.get("collection_name", None),
                "chunking_strategy": chunking_strategy,
                "embedding_model": embedding_model,
                "error": result.get("error_message", None)
            })
        
        # Process with Pinecone
        elif rag_method.lower() == "pinecone":
            # Initialize Pinecone
            client, _, index, _ = initialize_connections()
            
            # Use process_document_with_chunking instead of chunk_document
            chunks = process_document_with_chunking(markdown_content, chunking_strategy)
            
            # Format chunks for Pinecone
            formatted_chunks = [
                {"text": chunk, "file_path": markdown_path, "id": str(uuid.uuid4())} 
                for chunk in chunks
            ]
            
            # Generate and upload vectors
            vectors = prepare_vectors_for_upload(formatted_chunks, client)
            uploaded = upload_vectors_to_pinecone(vectors, index)
            
            # Update job status
            job_store[job_id].update({
                "status": "completed",
                "chunks_total": len(chunks),
                "vectors_uploaded": uploaded,
                "chunking_strategy": chunking_strategy
            })
        
        else:
            job_store[job_id].update({
                "status": "failed",
                "error": f"Unsupported RAG method: {rag_method}"
            })
            
    except Exception as e:
        api_logger.error(f"Error processing embeddings: {str(e)}")
        # Update job with error status
        job_store[job_id].update({
            "status": "failed",
            "error": str(e)
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

