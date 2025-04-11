from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional, Any
import os
import uuid
from Backend.logger import api_logger, pdf_logger, error_logger, log_request, log_error
import uvicorn
import sys
from pathlib import Path
import json

# Fix path handling for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)  # Add project root to path

# Local imports
from Backend.litellm_query_generator import generate_response, MODEL_CONFIGS
from Backend.parsing_methods.doclingparsing import main as docling_parse
from Backend.parsing_methods.mistralparsing import process_pdf as mistral_parse
from Backend.parsing_methods.mistralparsing_userpdf import process_pdf as mistral_parse_pdf  # Updated import

# Import modules with absolute paths
from Rag_modelings.chromadb_pipeline import (
    store_markdown_in_chromadb,
    query_and_generate_response as chromadb_query
)
from Rag_modelings.rag_pinecone import (
    query_pinecone_rag,  # Changed from run_rag_pipeline to query_pinecone_rag
    process_document_with_chunking,
    load_data_to_pinecone
)
from Rag_modelings.rag_manual import (
    process_document_with_chunking as manual_chunking,
    query_memory_rag,
    get_embedding as manual_get_embedding,
    initialize_anthropic_client
)


# Initialize FastAPI app
app = FastAPI(title="RAG Pipeline API",
              description="API for processing PDFs and querying the RAG system",
              version="1.0.0")
# Create routers for different functional groups
document_router = APIRouter(prefix="/documents", tags=["Document Processing"])
embedding_router = APIRouter(prefix="/rag", tags=["Embeddings & Vector Storage"])
query_router = APIRouter(prefix="/rag", tags=["RAG Queries"])
job_status_router = APIRouter(prefix="/status", tags=["Job Status"])
system_router = APIRouter(tags=["System"])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
BASE_DIR = "/app"  # for VM deployment
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MARKDOWN_DIR = os.path.join(BASE_DIR, "user_markdowns")
# Create directories with proper error handling and permissions
try:
    # Create directories if they don't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    
    # Set proper permissions (readable/writable by all)
    os.chmod(UPLOAD_DIR, 0o777)
    os.chmod(MARKDOWN_DIR, 0o777)
    
    print(f"Successfully created and set permissions for directories:")
    print(f"UPLOAD_DIR: {UPLOAD_DIR}")
    print(f"MARKDOWN_DIR: {MARKDOWN_DIR}")
except Exception as e:
    print(f"Error setting up directories: {str(e)}")
# In-memory storage
job_store = {}
query_job_store = {}
embedding_store = {}

class EmbeddingRequest(BaseModel):
    file_id: Optional[str] = None
    markdown_path: str
    markdown_filename: str
    rag_method: str  # "chromadb" or "pinecone"
    chunking_strategy: Optional[str] = "semantic_chunking"
    embedding_model: Optional[str] = "text-embedding-ada-002"
    similarity_metric: Optional[str] = "cosine"
    namespace: Optional[str] = None
    data_source: Optional[str] = None
    
class ManualEmbeddingRequest(BaseModel):
    text: str
    embedding_id: str
    rag_method: str  # "chromadb" or "pinecone"
    chunking_strategy: Optional[str]
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    embedding_id: Optional[str] = None  # For manual embeddings
    json_path: Optional[str] = None  # For Pinecone embeddings
    rag_method: str  # "chromadb" or "pinecone"
    data_source: Optional[str] = None  # "Nvidia Dataset" or "PDF Upload"
    quarters: Optional[List[str]] = None  # List of quarters to filter by
    model_id: str = "gpt-3.5-turbo"  # Changed from "gpt4o" to "gpt-3.5-turbo"
    similarity_metric: Optional[str] = "cosine"
    top_k: Optional[int] = 5
    namespace: Optional[str] = None
    

@system_router.get("/")
def read_root():
    return {"message": "RAG Pipeline API is running"}

@system_router.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available"}
@system_router.get("/llm/models",tags=["LLM Models"])
async def get_available_models():
    # Return a list of supported LLM models
    models = list(MODEL_CONFIGS.keys())
    api_logger.info(f"Available LLM models: {', '.join(models)}")
    return {"models": models}
@system_router.get("/nvidia/quarters",tags=["Nvidia Dataset Quarters"])
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

@document_router.post("/upload-and-parse",summary="Upload PDF and parse with selected parser")
async def upload_pdf(
    file: UploadFile = File(...),
    parser: str = Query("docling"),
):
    import tempfile
    temp_file = None
    try:
        file_id = str(uuid.uuid4())
        filename = file.filename
        
        # Create a temporary file instead of saving to UPLOAD_DIR
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file_path = temp_file.name
        temp_file.write(await file.read())
        temp_file.close()  # Close the file so it can be read by the parsers
        os.makedirs(MARKDOWN_DIR, exist_ok=True)
        # Generate the markdown path
        markdown_filename = os.path.splitext(filename)[0] + ".md"
        markdown_path = os.path.join(MARKDOWN_DIR, f"{file_id}_{markdown_filename}")
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)

        # Parse the PDF
        if parser.lower() == "docling":
            try:
                # Docling accepts file path
                result_path = docling_parse(temp_file_path)
                
                # Read the content from that file
                with open(result_path, "r", encoding="utf-8") as f:
                    parsed_content = f.read()
            except Exception as e:
                log_error(f"Error parsing PDF with Docling", e)
                parsed_content = f"# Error Parsing PDF\n\nFailed to parse {filename} with Docling: {str(e)}"
                
        elif parser.lower() == "mistral":
            try:
                # Mistral processing
                pdf_path = Path(temp_file_path)
                output_dir = Path(os.path.dirname(markdown_path))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Call Mistral with the file path
                result_path = mistral_parse_pdf(pdf_path, output_dir)
                if result_path and os.path.exists(result_path):
                    with open(result_path, "r", encoding="utf-8") as f:
                        parsed_content = f.read()
                        
                    if result_path != markdown_path:
                        with open(markdown_path, "w", encoding="utf-8") as f:
                            f.write(parsed_content)
                    api_logger.info(f"PDF parsed with Mistral: {result_path}")
                else:
                    # Handle the case where no result was returned
                    error_msg = "Mistral parser did not return a valid output path"
                    api_logger.error(error_msg)
                    parsed_content = f"# Error Parsing PDF\n\nFailed to parse {filename} with Mistral: {error_msg}"
            except Exception as e:
                log_error(f"Error parsing PDF with Mistral", e)
                parsed_content = f"# Error Parsing PDF\n\nFailed to parse {filename} with Mistral: {str(e)}"
        else:
            # Unsupported parser
            raise HTTPException(status_code=400, detail=f"Unsupported parser: {parser}")

        # Write the content to our standard location
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(parsed_content or "")

        return {
            "filename": filename,
            "file_id": file_id,
            "markdown_path": markdown_path,
            "markdown_filename": markdown_filename,
            "parser": parser,
            "status": "success"
        }

    except Exception as e:
        log_error(f"Error uploading PDF", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Delete temporary file if it exists
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
            api_logger.info(f"Temporary PDF file deleted: {temp_file.name}")
        

async def process_rag_query_task(
    query_job_id: str,
    query: str,
    rag_method: str,
    model_id: str,
    similarity_metric: str = "cosine",
    top_k: int = 5,
    namespace: str = None,
    json_path: str = None,
    data_source: str = None,
    quarters: List[str] = None,
    embedding_id: str = None
):
    try:
        api_logger.info(f"Starting background RAG query processing for job {query_job_id}")
        query_job_store[query_job_id]["status"] = "processing"
        
        result = None
        # Handle ChromaDB
        if rag_method.lower() == "chromadb":
            api_logger.info(f"Processing ChromaDB query: {query}")
            result = chromadb_query(
                query=query,
                similarity_metric=similarity_metric,
                llm_model=model_id,
                top_k=top_k,
                data_source=data_source,
                quarters=quarters
            )
            result["status"] = "completed"
        
        # Handle Pinecone
        elif rag_method.lower() == "pinecone":
            api_logger.info(f"Processing Pinecone query: {query}, with json_path: {json_path}")
            response = query_pinecone_rag(
                query=query,
                model_id=model_id,
                similarity_metric=similarity_metric,
                top_k=top_k,
                namespace=namespace,
                json_path=json_path
            )
            
            if "images_included" in response:
                response["has_images"] = response["images_included"]
            else:
                response["has_images"] = False
                
            # Format the result
            result = {
                "answer": response.get("answer", "Error generating response"),
                "usage": response.get("usage", {}),
                "source": "Pinecone",
                "chunks_used": response.get("chunks_used", 0),
                "has_images": response.get("has_images", False),
                "namespace": response.get("namespace", None),
                "status": "completed"
            }
        # Add this case to the existing function where it handles different RAG methods
        elif rag_method.lower() == "manual_embedding":
            api_logger.info(f"Processing manual embedding query: {query}")
            
            # Initialize client for manual RAG
            client = initialize_anthropic_client()
            
            # Use embedding_store if it exists, otherwise create a new one
            if embedding_id and embedding_id in embedding_store:
                embedding_data = embedding_store[embedding_id]
                memory_store = [{
                    "vector": manual_get_embedding(chunk, None),
                    "metadata": {"text": chunk, "text_preview": chunk[:100]}
                } for chunk in embedding_data["chunks"]]
            else:
                # For direct queries, create a simple memory store from the query itself
                api_logger.info("No existing embedding found, creating simple context")
                memory_store = [{
                    "vector": manual_get_embedding("This is a direct query with no context.", None),
                    "metadata": {
                        "text": "This is a direct query with no context.", 
                        "text_preview": "Direct query"
                    }
                }]
            
            # Use the query_memory_rag function from rag_manual.py
            response = query_memory_rag(
                query=query,
                memory_store=memory_store,
                client=client,
                top_k=top_k
            )
            
            result = {
                "answer": response.get("answer", "Error generating response"),
                "usage": {},  # Manual RAG doesn't track tokens currently
                "source": f"Manual embedding (ID: {embedding_id or 'direct_query'})",
                "chunks_used": len(memory_store),
                "status": "completed"
            }
        
        else:
            result = {
                "status": "failed",
                "error": f"Unsupported RAG method: {rag_method}"
            }
        
        # Update the query job store with the result
        query_job_store[query_job_id].update(result)
        api_logger.info(f"Completed RAG query processing for job {query_job_id}")
        
    except Exception as e:
        api_logger.error(f"Error processing RAG query in background task: {str(e)}")
        query_job_store[query_job_id].update({
            "status": "failed",
            "error": str(e)
        })

@query_router.post("/query",summary="Execute a RAG query with background processing")
async def rag_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Query the RAG system using the specified method and model with background processing"""
    try:
        log_request(f"RAG Query: {request.query}, Method: {request.rag_method}, Model: {request.model_id}, DataSource: {request.data_source}")
        log_request(f"Quarters: {request.quarters}")
        # Create a job ID for this query
        query_job_id = str(uuid.uuid4())
        
        # Initialize the query job in the store
        query_job_store[query_job_id] = {
            "status": "initializing",
            "query": request.query,
            "rag_method": request.rag_method,
            "model_id": request.model_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add the background task
        background_tasks.add_task(
            process_rag_query_task,
            query_job_id=query_job_id,
            query=request.query,
            rag_method=request.rag_method,
            model_id=request.model_id,
            similarity_metric=request.similarity_metric,
            top_k=request.top_k,
            namespace=request.namespace,
            json_path=request.json_path,
            data_source=request.data_source,
            quarters=request.quarters,
            embedding_id=request.embedding_id
        )
        
        # Return immediately with the job ID
        return {
            "query_job_id": query_job_id,
            "status": "processing",
            "message": "RAG query is being processed in the background",
            "poll_interval": 1,
            "status_endpoint": f"/query/{query_job_id}"
        }
        
    except Exception as e:
        log_error(f"Error initiating RAG query", e)
        raise HTTPException(status_code=500, detail=str(e))
    

@job_status_router.get("/query/{query_job_id}",summary="Get query job status")
async def get_query_status(query_job_id: str):
    """Get the status of a RAG query job"""
    if query_job_id not in query_job_store:
        raise HTTPException(status_code=404, detail="Query job not found")
    
    return query_job_store[query_job_id]

async def process_embeddings(
    job_id: str,
    markdown_path: str,
    markdown_filename: str,
    rag_method: str,
    chunking_strategy: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    similarity_metric: str = "cosine",
    namespace: str = None,
    data_source: str = None
):
    try:
        # Read markdown content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Extract file information
        file_name = markdown_filename
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
                source_info=source_info,
                similarity_metric=similarity_metric,
                data_source=data_source
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
            try:
                api_logger.info(f"Processing document with Pinecone using {chunking_strategy} strategy")
                
                # Use the streamlined function which handles the entire process
                result = load_data_to_pinecone(
                    markdown_content=markdown_content,
                    chunking_strategy=chunking_strategy,
                    file_name=file_name,
                    namespace=file_name,
                    similarity_metric=similarity_metric
                )
                
                if result["status"] == "success":
                    # Update job status with detailed information
                    job_store[job_id].update({
                        "status": "completed",
                        "chunks_total": result["total_chunks"],
                        "namespace": result["namespace"],
                        "vectors_uploaded": result["vectors_uploaded"],
                        "chunking_strategy": chunking_strategy,
                        "timestamp": datetime.now().isoformat(),
                        "file_name": file_name,
                        "json_path": result.get("json_path")
                    })
                    api_logger.info(f"Successfully processed document: {result['vectors_uploaded']} vectors uploaded and the chunks are stored in {result['json_path']}")
                else:
                    # Handle error case
                    job_store[job_id].update({
                        "status": "failed",
                        "error": result.get("error", "Unknown error"),
                        "timestamp": datetime.now().isoformat()
                    })
                    api_logger.error(f"Failed to process document: {result.get('error')}")
            except Exception as e:
                # Handle exceptions
                error_msg = f"Error processing document with Pinecone: {str(e)}"
                api_logger.error(error_msg)
                job_store[job_id].update({
                    "status": "failed",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
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

@embedding_router.post("/create-embeddings",summary="Create embeddings from a markdown file/nvidia datasets")
async def create_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """Create embeddings from a markdown file using ChromaDB or Pinecone"""
    try:
        log_request(f"Creating embeddings for {request.markdown_filename} using {request.rag_method}")
        
        # Verify the markdown file exists
        markdown_path = request.markdown_path
        markdown_filename = request.markdown_filename
        data_source = request.data_source
        # Create a job ID
        job_id = str(uuid.uuid4())
        similarity_metric = request.similarity_metric
        namespace = request.namespace
        
        # Initialize job status
        job_store[job_id] = {
            "status": "processing",
            "markdown_path": markdown_path,
            "rag_method": request.rag_method,
            "chunking_strategy": request.chunking_strategy,
            "similarity_metric": similarity_metric,
            "namespace": namespace
        }
        
        # Process embeddings in the background
        background_tasks.add_task(
            process_embeddings,
            job_id,
            markdown_path,
            markdown_filename,
            request.rag_method,
            request.chunking_strategy,
            request.embedding_model,
            similarity_metric,
            namespace,
            data_source
        )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"Creating embeddings for {os.path.basename(markdown_path)} using {request.rag_method}",
            "poll_interval": 2,  # Poll every 2 seconds
            "status_endpoint": f"/job/{job_id}"  # Provide the endpoint to poll
        }
    except Exception as e:
        log_error(f"Error creating embeddings", e)
        raise HTTPException(status_code=500, detail=str(e))
    
@embedding_router.post("/manual-embedding", summary="Create embeddings from text provided directly by the user")
async def create_manual_embedding(request: ManualEmbeddingRequest):
    """Create embeddings from text provided directly by the user"""
    try:
        log_request(f"Creating manual embedding with ID {request.embedding_id}")
        
        # Set default metadata if not provided
        metadata = request.metadata or {
            "source": "manual_input",
            "timestamp": datetime.now().isoformat()
        }
        
        # Apply chunking using rag_manual's function
        chunks = manual_chunking(request.text, request.chunking_strategy)
        
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

@job_status_router.get("/job/{job_id}",summary="Get embedding job status")
async def get_job_status(job_id: str):
    """Get the status of an embedding job"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_store[job_id]

@embedding_router.get("/embeddings",summary="List all manual embeddings stored in memory")
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

@embedding_router.post("/config",summary="Set the RAG configuration")
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
    




# Include all routers
app.include_router(system_router)
app.include_router(document_router)
app.include_router(embedding_router)
app.include_router(query_router)
app.include_router(job_status_router)

