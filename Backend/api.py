from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional
import os
import uuid
import asyncio
from logger import api_logger, pdf_logger, error_logger, log_request, log_error

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
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory job storage
job_store = {}

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
    parser: Optional[str] = Query(None),
    rag_method: Optional[str] = Query(None),
    chunking_strategy: Optional[str] = Query(None)
):
    try:
        log_request(f"PDF Upload: {file.filename}, Parser: {parser}, RAG: {rag_method}, Chunking: {chunking_strategy}")
        
        # Create a unique filename
        file_id = f"{uuid.uuid4()}"
        file_extension = os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        pdf_logger.info(f"File saved to {file_path}")
        
        # Return success response
        return {
            "filename": file.filename,
            "saved_as": file_path,
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
    models = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus", "mistral-7b"]
    return {"models": models}

@app.post("/rag/query")
async def rag_query(request: RAGRequest, background_tasks: BackgroundTasks):
    try:
        log_request(f"RAG Query: {request.question}, Model: {request.model}, Quarters: {request.quarters}")
        
        # Initialize the job with a pending status
        job_store[request.request_id] = {
            "status": "processing",
            "question": request.question,
            "model": request.model
        }
        
        # Process the query in the background
        background_tasks.add_task(
            process_rag_query, 
            request.request_id, 
            request.question, 
            request.model,
            request.parser,
            request.rag_method,
            request.chunking_strategy,
            request.quarters
        )
        
        return {"request_id": request.request_id, "status": "submitted"}
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

# Background processing function
async def process_rag_query(
    request_id: str, 
    question: str, 
    model: str,
    parser: Optional[str], 
    rag_method: Optional[str], 
    chunking_strategy: Optional[str],
    quarters: Optional[List[str]]
):
    try:
        # Simulate processing time
        await asyncio.sleep(3)
        
        # In a real implementation, this would:
        # 1. Call the appropriate RAG implementation based on configuration
        # 2. Get chunks from the vector store
        # 3. Send to LLM and format response
        
        # Update job with completed results
        job_store[request_id] = {
            "status": "completed",
            "answer": f"This is a simulated answer to: '{question}' using {model}.",
            "sources": [
                {"document": f"NVIDIA-{q}", "text": f"Sample text from {q} report"} 
                for q in (quarters or ["2023q4"])
            ]
        }
        
        api_logger.info(f"Completed processing request {request_id}")
    except Exception as e:
        log_error(f"Error in background RAG processing for {request_id}", e)
        # Update job with error status
        job_store[request_id] = {
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)