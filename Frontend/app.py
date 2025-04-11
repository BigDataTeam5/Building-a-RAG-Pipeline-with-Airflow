import streamlit as st
import requests
import time
import sys
import os
import json
import uuid
import toml
import re

# Streamlit UI
st.set_page_config(page_title="Nvidia Quarterly data RAG", layout="wide")
current_page = st.query_params.get("page", "main")

if "show_token_usage" not in st.session_state:
    st.session_state.show_token_usage = False
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
# Initialize session state for LLM responses
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "question_result" not in st.session_state:
    st.session_state.question_result = None
if "processing_summary" not in st.session_state:
    st.session_state.processing_summary = False
if "processing_question" not in st.session_state:
    st.session_state.processing_question = False
if "similarity_metric" not in st.session_state:
    st.session_state.similarity_metric = False
# RAG-specific session states
if "pdf_parser" not in st.session_state:
    st.session_state.pdf_parser = None
if "rag_method" not in st.session_state:
    st.session_state.rag_method = None
if "chunking_strategy" not in st.session_state:
    st.session_state.chunking_strategy = None
if "selected_quarters" not in st.session_state:
    st.session_state.selected_quarters = []
if "namespace" not in st.session_state:
    st.session_state.namespace = None
if "embedding_response" not in st.session_state:
    st.session_state.embedding_response = {}

# FastAPI Base URL - Simple configuration
if "fastapi_url" not in st.session_state:
    config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secret.toml")
    if os.path.exists(config_path):
        config_data = toml.load(config_path)
        st.session_state.fastapi_url = config_data.get("connections", {}).get("FASTAPI_URL")
                
if "api_connected" not in st.session_state:
    st.session_state.api_connected = True


def update_api_endpoints():
    base_url = st.session_state.fastapi_url
    
    # API Endpoints - updated to match new router structure
    st.session_state.UPLOAD_PDF_PARSE_API = f"{base_url}/documents/upload-and-parse"  
    st.session_state.RAG_EMBED_API = f"{base_url}/rag/create-embeddings"
    st.session_state.RAG_MANUAL_EMBED_API = f"{base_url}/rag/manual-embedding"
    st.session_state.LLM_MODELS_API = f"{base_url}/llm/models"
    st.session_state.NVIDIA_QUARTERS_API = f"{base_url}/nvidia/quarters"
    st.session_state.RAG_QUERY_API = f"{base_url}/rag/query"
    st.session_state.RAG_CONFIG_API = f"{base_url}/rag/config"
    st.session_state.RAG_JOB_STATUS_API = f"{base_url}/status/job"
    st.session_state.RAG_QUERY_STATUS_API = f"{base_url}/status/query"

# Initial setup of API endpoints
update_api_endpoints()

# Function to Upload File - With updated parameters to only use PDF file and parser method
def upload_pdf(file):
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        with st.spinner("📤 Uploading PDF... Please wait."):
            # Include only parser parameter
            params = {
                "parser": st.session_state.pdf_parser
            }
            response = requests.post(st.session_state.UPLOAD_PDF_PARSE_API, files=files, params=params)

        if response.status_code == 200:
            st.session_state.file_uploaded = True
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", f"Upload failed: {response.status_code}")
            except ValueError:
                error_detail = f"Upload failed with status {response.status_code}: {response.text}"
            st.error(f"Error: {error_detail}")
            return {"error": error_detail}
    except requests.RequestException as e:
        st.error(f"Request Exception: {str(e)}")
        return {"error": str(e)}

# Function to fetch available quarters from the Nvidia dataset
def fetch_nvidia_quarters():
    try:
        response = requests.get(st.session_state.NVIDIA_QUARTERS_API)
        if response.status_code == 200:
            return response.json().get("quarters", [])
        else:
            st.warning(f"Could not fetch available quarters: {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Error fetching quarters: {str(e)}")
        return []

def submit_rag_query(question, model, quarters=None, json_path=None):
    try:
        # Prepare the payload based on RAG method
        if st.session_state.rag_method == "Pinecone":
            payload = {
                "query": question,
                "rag_method": "pinecone",
                "model_id": model,
                "similarity_metric": st.session_state.similarity_metric,
                "namespace": st.session_state.get("namespace"),
                "json_path": json_path
            }
        elif st.session_state.rag_method.lower() == "chromadb":
            if st.session_state.data_source == "Nvidia Dataset" and quarters:
                
                payload = {
                    "query": question,
                    "rag_method": "chromadb",
                    "model_id": model,
                    "data_source": st.session_state.data_source,
                    "quarters": quarters
                }
            else:
                payload = {
                    "query": question,
                    "rag_method": "chromadb",
                    "model_id": model,
                    "data_source": st.session_state.data_source,
                    "json_path": json_path
                }
        elif st.session_state.rag_method.lower() == "manual_embedding":
            payload = {
                "query": question,
                "rag_method": "manual_embedding",
                "model_id": model,
                "embedding_id": st.session_state.get("manual_embedding_id","direct_query")
            }
        else:
            payload = {
                "query": question,
                "rag_method": st.session_state.rag_method.lower(),
                "model_id": model,
                "data_source": st.session_state.data_source
            }
        
        # Submit the query to the API
        response = requests.post(
            st.session_state.RAG_QUERY_API,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if "query_job_id" in result:
                st.info("Query is being processed in the background. Please wait...")
                final_result = poll_for_query_status(result["query_job_id"])
                return final_result
            else:
                return result
        else:
            st.error(f"Failed to submit query: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error in query processing: {str(e)}")
        return None

# Function to fetch available LLM models from API
def fetch_available_models():
    """Fetch available LLM models from the backend API"""
    try:
        response = requests.get(st.session_state.LLM_MODELS_API)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return models
        else:
            st.warning(f"Could not fetch available models: {response.status_code}")
            return ["gpt-3.5-turbo"]  # Default fallback model
    except Exception as e:
        st.warning(f"Error fetching models: {str(e)}")
        return ["gpt-3.5-turbo"]  # Default fallback model
    



def poll_for_query_status(query_job_id, interval=1):
    """Poll for RAG query job status with dynamic progress indicators"""
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_msg = st.empty()
    start_time = time.time()
    cycle_duration = 60  # Reset progress visual every 60 seconds
    
    # Use a spinner for additional visual cue
    with st.spinner("Processing your question with RAG pipeline..."):
        while True:
            try:
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Create a cycle for the progress indicator
                cycle_progress = (elapsed_time % cycle_duration) / cycle_duration
                
                # Update progress bar
                progress_bar.progress(min(cycle_progress, 0.97))
                
                # Update status message with more detailed information
                elapsed_mins = int(elapsed_time // 60)
                elapsed_secs = int(elapsed_time % 60)
                
                # Dynamic status messages
                phase = int(elapsed_time % 12)
                if phase < 3:
                    status_text = "Retrieving relevant chunks..."
                elif phase < 6:
                    status_text = "Processing query with LLM..."
                elif phase < 9:
                    status_text = "Generating comprehensive response..."
                else:
                    status_text = "Finalizing answer..."
                
                status_msg.text(f"⏳ {status_text} (Elapsed: {elapsed_mins}m {elapsed_secs}s)")
                
                query_status_url = f"{st.session_state.RAG_QUERY_STATUS_API}/{query_job_id}"
                response = requests.get(
                    query_status_url,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    status = result_data.get("status")
                    
                    if status == "completed":
                        progress_bar.progress(1.0)
                        status_msg.text("✅ Response generated successfully!")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_msg.empty()
                        return result_data
                        
                    elif status == "failed":
                        progress_bar.empty()
                        status_msg.empty()
                        st.error(f"Query processing failed: {result_data.get('error', 'Unknown error')}")
                        return None
                
                # Sleep before next check
                time.sleep(interval)
                
            except Exception as e:
                progress_bar.empty()
                status_msg.empty()
                st.error(f"Error polling for query status: {str(e)}")
                return None

# Add this token usage calculation function
def calculate_token_cost(model_id, usage_data):
    """Calculate cost based on model and token usage"""
    # Default rates (can be adjusted based on actual pricing)
    rates = {
        "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
        "gemini": {"input": 0.000001, "output": 0.000002},
        "deepseek": {"input": 0.000003, "output": 0.000005},
        "claude": {"input": 0.000025, "output": 0.000075},
        "grok": {"input": 0.000004, "output": 0.000006},
        "default": {"input": 0.000002, "output": 0.000004}
    }
    
    # Get rates for the model or use default
    model_rates = rates.get(model_id.lower(), rates["default"])
    
    # Calculate costs
    input_cost = (usage_data.get("prompt_tokens", 0) * model_rates["input"])
    output_cost = (usage_data.get("completion_tokens", 0) * model_rates["output"])
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# Fetch models at startup
if "available_models" not in st.session_state:
    st.session_state.available_models = fetch_available_models()
# Function to process RAG embeddings
def process_rag_embeddings(file_id, markdown_path, markdown_filename, rag_method, chunking_strategy,similarity_metric):
    try:
        # Prepare the request payload
        if rag_method.lower() == "pinecone":
            namespace = markdown_filename.lower().replace(" ", "-").replace(".", "-")
            # Store the namespace in session state for future queries
            st.session_state.namespace = namespace
            payload = {
            "file_id": file_id,
            "markdown_path": markdown_path,
            "markdown_filename": markdown_filename,
            "rag_method": rag_method,
            "chunking_strategy": chunking_strategy,
            "similarity_metric": similarity_metric,
            "namespace": namespace
        }
        else:
            payload = {
            "file_id": file_id,
            "markdown_path": markdown_path,
            "markdown_filename": markdown_filename,
            "rag_method": rag_method,
            "chunking_strategy": chunking_strategy,
            "similarity_metric": similarity_metric
        }       
        st.session_state.markdown_filename = markdown_filename
        
        # Make the API request
        response = requests.post(
            st.session_state.RAG_EMBED_API,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Failed to process embeddings: {response.text}")
            return {"error": response.text}
    except Exception as e:
        st.error(f"Error processing embeddings: {str(e)}")
        return {"error": str(e)}
        
def poll_for_embedding_status(job_id, interval=2):
    """Poll for embedding job status with indefinite waiting and dynamic progress indicators"""
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_msg = st.empty()
    error_msg = st.empty()  # Add an error message placeholder
    start_time = time.time()
    cycle_duration = 90  # Reset progress visual every 90 seconds for continued engagement
    
    # Track consecutive errors to prevent infinite polling on terminal failures
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    # Use a spinner for additional visual cue
    with st.spinner("Creating embeddings and storing vectors in database..."):
        while True:
            try:
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Create a cycle for the progress indicator (resets periodically)
                cycle_progress = (elapsed_time % cycle_duration) / cycle_duration
                
                # Update progress bar
                progress_bar.progress(min(cycle_progress, 0.97))  # Cap at 97% until complete
                
                # Update status message with more detailed information
                elapsed_mins = int(elapsed_time // 60)
                elapsed_secs = int(elapsed_time % 60)
                
                # Dynamic status messages based on RAG method and elapsed time
                if st.session_state.rag_method == "Pinecone":
                    phase = elapsed_time % 9  # Cycle through 3 phases every 9 seconds
                    if phase < 3:
                        status_text = "Processing document chunks..."
                    elif phase < 6:
                        status_text = "Creating vector embeddings..."
                    else:
                        status_text = "Loading vectors to Pinecone..."
                elif st.session_state.rag_method == "ChromaDB":
                    phase = elapsed_time % 9
                    if phase < 3:
                        status_text = "Processing document chunks..."
                    elif phase < 6:
                        status_text = "Creating vector embeddings..."
                    else:
                        status_text = "Loading embeddings to ChromaDB..."
                elif st.session_state.rag_method == "manual_embedding":
                    phase = elapsed_time % 6
                    if phase < 3:
                        status_text = "Processing document chunks..."
                    else:
                        status_text = "Creating manual embeddings in memory..."
                else:
                    phase = elapsed_time % 9
                    if phase < 3:
                        status_text = "Processing document chunks..."
                    elif phase < 6:
                        status_text = "Creating vector embeddings..."
                    else:
                        status_text = "Storing vectors in database..."
                
                status_msg.text(f"⏳ {status_text} (Elapsed: {elapsed_mins}m {elapsed_secs}s)")
                
                
                job_status_url = f"{st.session_state.RAG_JOB_STATUS_API}/{job_id}"
                
                # Increase timeout and implement retries
                max_retries = 3
                retry_count = 0
                response = None
                
                while retry_count < max_retries:
                    try:
                        response = requests.get(
                            job_status_url,
                            timeout=60  # Increased timeout for the request
                        )
                        # Reset consecutive errors counter on successful connection
                        consecutive_errors = 0
                        break  # Exit retry loop if successful
                    except requests.exceptions.Timeout:
                        retry_count += 1
                        if retry_count < max_retries:
                            # Log timeout and retry
                            status_msg.text(f"⏳ Connection timeout, retrying... (Attempt {retry_count}/{max_retries})")
                            time.sleep(2)  # Wait before retry
                        else:
                            # If we reach max retries but the process is still running
                            status_msg.text(f"⏳ Operation taking longer than expected, but still running... (Elapsed: {elapsed_mins}m {elapsed_secs}s)")
                            consecutive_errors += 1
                    except Exception as e:
                        # For other exceptions, try again but don't fail completely
                        retry_count += 1
                        if retry_count < max_retries:
                            status_msg.text(f"⏳ Connection error, retrying... (Attempt {retry_count}/{max_retries})")
                            time.sleep(2)
                        else:
                            consecutive_errors += 1
                
                # If we've hit too many consecutive errors, provide a way to continue
                if consecutive_errors >= max_consecutive_errors:
                    error_msg.error("Multiple connection errors. The server may be unreachable or the operation may have failed.")
                    
                    # Give user option to continue trying or abort
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Continue Trying"):
                            consecutive_errors = 0
                            error_msg.empty()
                    with col2:
                        if st.button("Abort"):
                            progress_bar.empty()
                            status_msg.empty()
                            error_msg.empty()
                            return {"status": "aborted", "error": "User aborted after multiple connection errors"}
                
                # If we have a response, process it
                if response and response.status_code == 200:
                    result_data = response.json()
                    status = result_data.get("status")
                    
                    if status == "completed":
                        progress_bar.progress(1.0)  
                        
                        # Success message based on RAG method
                        if st.session_state.rag_method == "Pinecone":
                            status_msg.text("✅ Embeddings successfully loaded to Pinecone!")
                            if "json_path" in result_data:
                                st.session_state.namespace = result_data.get("json_path")
                            if "namespace" in result_data:
                                st.session_state.namespace = result_data.get("namespace")
                            if "json_path" in result_data:
                                st.session_state.json_path = result_data.get("json_path")
                        elif st.session_state.rag_method == "ChromaDB":
                            status_msg.text("✅ Embeddings successfully loaded to ChromaDB!")
                        elif st.session_state.rag_method == "manual_embedding":
                            status_msg.text("✅ Manual embeddings created successfully in memory!")
                        else:
                            status_msg.text("✅ Embeddings created successfully!")
                            
                        time.sleep(0.5)  # Brief pause to show completed progress
                        progress_bar.empty()  # Remove the progress bar
                        status_msg.empty()  # Remove the status message
                        error_msg.empty()   # Remove error message if any
                        return result_data
                        
                    elif status == "failed":
                        progress_bar.empty()
                        status_msg.empty()
                        error_msg.empty()
                        st.error(f"Embedding creation failed: {result_data.get('error', 'Unknown error')}")
                        return None
                
                # Sleep before next check
                time.sleep(interval)
                
            except Exception as e:
                # Log error but don't immediately fail
                status_msg.text(f"⚠️ Error checking status: {str(e)}. Will retry... (Elapsed: {elapsed_mins}m {elapsed_secs}s)")
                consecutive_errors += 1
                time.sleep(interval * 2)  # Wait longer before retrying
                continue  # Continue polling instead of stopping with an error
# Sidebar UI
with st.sidebar:
    st.title("RAG Pipeline Configuration")
    
    # Dropdown for data source
    data_source = st.selectbox(
        "Select Data Source:",
        ["Select an option", "PDF Upload", "Nvidia Dataset"],
        index=0
    )
    similarity_metric = st.selectbox(
            "Select Similarity Metric:",
            [   "Select a metric",
                "cosine",
                "euclidean",
                "dot-product" 
            ],
            help="Choose how similarity between text chunks will be calculated"
        )
    
    # Only show these options for PDF Upload
    if data_source == "PDF Upload" and similarity_metric!="Select a metric":
        st.subheader("Configuration Options")
        # PDF parser selection
        pdf_parser = st.selectbox(
            "Select PDF Parser:",
            ["Select a parser", "Docling", "Mistral"],
            index=0
        )
        
        # RAG method selection
        rag_method = st.selectbox(
            "Select RAG Method:",
            ["Select a method", "manual_embedding", "Pinecone", "ChromaDB"],
            index=0
        )
        
        # Chunking strategy
        chunking_strategy = st.selectbox(
            "Select Chunking Strategy:",
            ["Select a strategy", "characterbased_chunking","recursive_chunking", "semantic_chunking"],
            index=0
        )
        
        # LLM model selection
        available_models = ["Select Model"] + st.session_state.available_models
        llm_model = st.selectbox("Select LLM Model:", available_models, index=0)
        
        # Save selections to session state
        st.session_state.pdf_parser = pdf_parser if pdf_parser != "Select a parser" else None
        st.session_state.rag_method = rag_method if rag_method != "Select a method" else None
        st.session_state.chunking_strategy = chunking_strategy if chunking_strategy != "Select a strategy" else None
        st.session_state.llm_model = llm_model if llm_model != "Select Model" else None
    
    # For Nvidia Dataset, set default values
    elif data_source == "Nvidia Dataset" and similarity_metric!="Select a metric":
        st.subheader("Configuration Options")
        # Set default values for Nvidia Dataset
        st.session_state.pdf_parser = "Mistral"  
        st.session_state.rag_method = "ChromaDB"  
        st.session_state.chunking_strategy = "semantic_chunking"  
        st.session_state.llm_model = st.session_state.available_models[0] if st.session_state.available_models else "gpt-3.5-turbo"  # Default model
        
        # Display info about default configuration
        st.info("**Default Configuration for Nvidia Dataset**")
        st.info(f"• PDF Parser: {st.session_state.pdf_parser}")
        st.info(f"• RAG Method: {st.session_state.rag_method}")
        st.info(f"• Chunking: {st.session_state.chunking_strategy}")
        
        # Show quarter selection
        available_quarters = fetch_nvidia_quarters()
        if available_quarters:
            selected_quarters = st.multiselect(
                "Select Quarters to Query:",
                available_quarters,
                default=available_quarters[:1] if available_quarters else []
            )
            st.session_state.selected_quarters = selected_quarters
    
    # Save data source to session state
    st.session_state.data_source = data_source
    
    # Apply configuration button
    if st.button("Apply Configuration"):
        if data_source == "Select an option":
            st.error("Please select a data source")
        elif data_source == "PDF Upload":
            if any(val == None or "Select" in str(val) for val in [
                st.session_state.pdf_parser, 
                st.session_state.rag_method, 
                st.session_state.chunking_strategy, 
                st.session_state.llm_model
            ]):
                st.error("Please select all configuration options")
            else:
                st.session_state.next_clicked = True
                st.success("Configuration applied successfully")
        elif data_source == "Nvidia Dataset" and not selected_quarters:
            st.error("Please select at least one quarter")
        else:
            st.session_state.next_clicked = True
            st.success("Configuration applied successfully")
    st.markdown("---")
    st.subheader("Analytics")
    if st.button("📊 View Token Usage"):
        st.query_params["page"] = "token_usage"

if current_page == "token_usage":
    # Token usage page
    st.title("📊 Token Usage Analytics")
    
    # Back button using query params
    if st.button("🔙 Back to Main Page"):
        st.query_params["page"] = "main"
        
    st.markdown("### Token Usage History")
    if "token_usage_records" not in st.session_state:
        st.session_state.token_usage_records = []
        
    if hasattr(st.session_state, "question_result") and st.session_state.question_result:
        result = st.session_state.question_result
        
        # Extract usage data with better error handling
        usage_data = {}
        
        # Handle different response formats
        if "usage" in result:
            usage_data = result.get("usage", {})
            
            # Handle string format (JSON string)
            if isinstance(usage_data, str):
                try:
                    usage_data = json.loads(usage_data)
                except json.JSONDecodeError:
                    st.warning("Could not parse usage data")
                    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            # Handle missing fields
            if "prompt_tokens" not in usage_data or "completion_tokens" not in usage_data:
                # Try to estimate from content if available
                if "answer" in result and result["answer"]:
                    answer_length = len(result["answer"])
                    estimated_tokens = answer_length // 4  # Rough estimate of tokens
                    
                    if "prompt_tokens" not in usage_data:
                        usage_data["prompt_tokens"] = estimated_tokens * 4  # Typical prompt/completion ratio
                    
                    if "completion_tokens" not in usage_data:
                        usage_data["completion_tokens"] = estimated_tokens
                    
                    if "total_tokens" not in usage_data:
                        usage_data["total_tokens"] = usage_data.get("prompt_tokens", 0) + usage_data.get("completion_tokens", 0)
        
        # If we still have no usage data, create default values with warning
        if not usage_data:
            st.warning("No token usage data available. Using estimated values.")
            # Set default values based on length of response
            answer_text = result.get("answer", "No answer")
            usage_data = {
                "prompt_tokens": max(len(result.get("query", "")) // 3, 10),
                "completion_tokens": len(answer_text) // 4,
                "total_tokens": (len(result.get("query", "")) // 3) + (len(answer_text) // 4)
            }
    
        # Calculate costs based on the model - MOVED INSIDE IF BLOCK
        model_id = result.get("model_id", "gpt-3.5-turbo")
        cost_data = calculate_token_cost(model_id, usage_data)
        
        # Create a timestamp if not present - MOVED INSIDE IF BLOCK
        if "timestamp" not in result:
            result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
        # Add to usage records if not already there - MOVED INSIDE IF BLOCK
        query_id = result.get("job_id", str(uuid.uuid4()))
        
        # Check if this exact query is already recorded - MOVED INSIDE IF BLOCK
        if not any(record.get("job_id") == query_id for record in st.session_state.token_usage_records):
            st.session_state.token_usage_records.append({
                "job_id": query_id,
                "task_type": "RAG Query",
                "query": result.get("query", "")[:30] + "..." if len(result.get("query", "")) > 30 else result.get("query", ""),
                "model": model_id,
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
                "cost": cost_data["total_cost"],
                "timestamp": result["timestamp"]
            })
    
    # Display usage statistics
    if not st.session_state.token_usage_records:
        st.info("No token usage data available yet. Ask questions to see usage statistics.")
    else:
        # Calculate total tokens and cost
        # This section doesn't use the result variable, so it's fine outside the if block
        total_tokens = sum(record["total_tokens"] for record in st.session_state.token_usage_records)
        total_cost = sum(record["cost"] for record in st.session_state.token_usage_records)
        prompt_tokens = sum(record["prompt_tokens"] for record in st.session_state.token_usage_records)
        completion_tokens = sum(record["completion_tokens"] for record in st.session_state.token_usage_records)
        
        # Display overall metrics
        st.subheader("Overall Usage")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tokens", f"{total_tokens:,}")
        col2.metric("Input Tokens", f"{prompt_tokens:,}")
        col3.metric("Output Tokens", f"{completion_tokens:,}")
        col4.metric("Total Cost", f"${total_cost:.5f}")
        
        # Display detailed usage table
        st.subheader("Detailed Usage History")
        
        # Convert to DataFrame for display
        import pandas as pd
        df = pd.DataFrame(st.session_state.token_usage_records)
        
        # Reorder columns for better display
        column_order = ["timestamp", "task_type", "query", "model", "prompt_tokens", 
                        "completion_tokens", "total_tokens", "cost"]
        
        # Filter columns that actually exist in the dataframe
        column_order = [col for col in column_order if col in df.columns]
        
        # Display dataframe with formatted columns
        st.dataframe(df[column_order].style.format({
            "cost": "${:.5f}",
            "prompt_tokens": "{:,}",
            "completion_tokens": "{:,}",
            "total_tokens": "{:,}"
        }), use_container_width=True)
        
        # Breakdown by model if we have multiple models
        model_counts = df["model"].nunique()
        if model_counts > 1:
            st.subheader("Usage by Model")
            model_usage = df.groupby("model").agg({
                "prompt_tokens": "sum",
                "completion_tokens": "sum", 
                "total_tokens": "sum",
                "cost": "sum"
            }).reset_index()
            
            st.dataframe(model_usage.style.format({
                "cost": "${:.5f}",
                "prompt_tokens": "{:,}",
                "completion_tokens": "{:,}",
                "total_tokens": "{:,}"
            }), use_container_width=True)
            
            # Visualization of token usage
            try:
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # Token distribution
                ax1.pie(model_usage["total_tokens"], labels=model_usage["model"], autopct='%1.1f%%')
                ax1.set_title("Token Distribution by Model")
                
                # Cost distribution
                ax2.pie(model_usage["cost"], labels=model_usage["model"], autopct='%1.1f%%')
                ax2.set_title("Cost Distribution by Model")
                
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate charts: {str(e)}")

else:  # current_page == "main" or any other value
    # Regular main page content
    st.title("📄 RAG Pipeline with Airflow")

    if st.session_state.get("next_clicked", False):
        if st.session_state.data_source == "PDF Upload":
            st.header("Upload PDF Document")
            
            # Display file uploader
            uploaded_file = st.file_uploader("Upload a PDF File:", type=["pdf"], key="pdf_uploader")
            if uploaded_file:
                st.session_state.file_uploaded = True
                upload_response = upload_pdf(uploaded_file)
                if "error" not in upload_response:
                    st.success("✅ PDF File Uploaded Successfully!")
                    st.info("The document will be processed through the RAG pipeline with your selected configuration.")
                    
                    # Display embedding button after successful upload
                    if "file_id" in upload_response and "markdown_path" in upload_response:
                        file_id = upload_response["file_id"]
                        markdown_path = upload_response["markdown_path"]
                        markdown_filename = upload_response["markdown_filename"]
                        
        
                    if st.button("Process Chunking & Create Embeddings"):
                        with st.spinner("Processing embeddings..."):
                            if st.session_state.rag_method == "manual_embedding":
                                try:
                                    with open(markdown_path, "r", encoding="utf-8") as f:
                                        markdown_content = f.read()
                                    try:
                                        with open(markdown_path, "r", encoding="utf-8") as f:
                                            markdown_content = f.read()
                                    except FileNotFoundError:
                                        st.error(f"File not found: {markdown_path}")
                                        st.info("Please upload a PDF file first")
                                        st.stop()

                                    # Create a unique ID for this manual embedding
                                    embedding_id = f"manual_{file_id}"
                                    
                                    # Call the manual embedding endpoint
                                    manual_embed_response = requests.post(
                                        st.session_state.RAG_MANUAL_EMBED_API,
                                        json={
                                            "text": markdown_content,
                                            "embedding_id": embedding_id,
                                            "rag_method": "manual_embedding",
                                            "chunking_strategy": st.session_state.chunking_strategy,
                                            "metadata": {
                                                "file_name": markdown_filename,
                                                "source": "pdf_upload"
                                            }
                                        }
                                    )
                                    
                                    if manual_embed_response.status_code == 200:
                                        result = manual_embed_response.json()
                                        st.session_state.manual_embedding_id = embedding_id
                                        st.session_state.embedding_response = {
                                            "status": "completed",
                                            "job_id": embedding_id,
                                            "chunks_count": result.get('chunks_count', 0)
                                        }
                                        st.success("✅ Manual embeddings created successfully!")
                                        st.info(f"Created {result.get('chunks_count', 0)} chunks in memory")
                                    else:
                                        st.session_state.embedding_response = {"error": manual_embed_response.text}
                                        st.error(f"Manual embedding failed: {manual_embed_response.text}")
                                except Exception as e:
                                    st.session_state.embedding_response = {"error": str(e)}
                                    st.error(f"Error creating manual embeddings: {str(e)}")
                            else:
                                # Original code for other RAG methods
                                response = process_rag_embeddings(
                                    file_id, 
                                    markdown_path,
                                    markdown_filename,
                                    st.session_state.rag_method,
                                    st.session_state.chunking_strategy,
                                    similarity_metric
                                )
                                st.session_state.embedding_response = response
                                
                                # Handle polling for non-manual embeddings
                                if "error" not in response:
                                    job_id = response.get("job_id")
                                    if job_id:
                                        final_result = poll_for_embedding_status(job_id)
                                        
                                        if final_result and final_result.get("status") == "completed":
                                            st.session_state.embedding_job_id = job_id
                                            st.success("✅ Embeddings created successfully!")
                                            namespace = markdown_filename.lower().replace(" ", "-").replace(".", "-")
                                            st.info(f"Your document was indexed in namespace: **{namespace}**")
                                        else:
                                            st.error("Failed to create embeddings. Please try again.")
                                    else:
                                        st.error("No job ID returned from the embedding creation process.")
                                elif response:  # Only show error if we have a response
                                    st.error(f"Embedding failed: {response.get('error')}")

        elif st.session_state.data_source == "Nvidia Dataset":
            st.header("Query Nvidia Quarterly Reports")
            
            # Display selected configuration
            st.subheader("Current Configuration:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**PDF Parser:** {st.session_state.pdf_parser}")
            with col2:
                st.info(f"**RAG Method:** {st.session_state.rag_method}")
            with col3:
                st.info(f"**Chunking Strategy:** {st.session_state.chunking_strategy}")
            
            # Display selected quarters
            st.subheader("Selected Quarters:")
            if st.session_state.selected_quarters:
                quarters_display = ", ".join(st.session_state.selected_quarters)
                st.info(f"Querying data from: **{quarters_display}**")
            else:
                st.warning("No quarters selected. Please select at least one quarter in the sidebar.")
            
        # Question and answer section (common for both data sources)
        if ((st.session_state.data_source == "PDF Upload" and st.session_state.file_uploaded) or 
            (st.session_state.data_source == "Nvidia Dataset" and st.session_state.selected_quarters)):
            
            st.markdown("---")
            st.subheader("Ask Questions About the Data")
            # Similarity metric selection
            
            st.session_state.similarity_metric = similarity_metric
            
            user_question = st.text_area(
                "Enter your question:",
                placeholder="Example: What was Nvidia's revenue in Q2 2023?",
                key="rag_question"
            )
            
            if st.button("Submit Query", type="primary"):
                if not user_question:
                    st.error("Please enter a question.")
                else:
                    # Submit RAG query using selected configuration
                    quarters = st.session_state.selected_quarters if st.session_state.data_source == "Nvidia Dataset" else None
                    json_path = st.session_state.get("json_path")
                    result = submit_rag_query(
                        question=user_question, 
                        model=st.session_state.llm_model, 
                        quarters=quarters,
                        json_path=json_path
                        )
                    
                    if result:
                        # Display the answer
                        st.session_state.question_result = result
                        
                        # Show the answer with citations
                        st.markdown("### Answer")
                        answer_text = result.get("answer") or result.get("response") or "No answer available"
                        st.markdown(answer_text)
                        
                        # Show source documents and match information
                        if "matches" in result:
                            with st.expander("Source Documents & Match Information", expanded=False):
                                for idx, match in enumerate(result["matches"]):
                                    st.markdown(f"### Match {idx+1}")
                                    
                                    # Display match score
                                    st.markdown(f"**Similarity Score:** {match.get('score', 'N/A'):.4f}")
                                    
                                    # Display metadata
                                    metadata = match.get('metadata', {})
                                    if metadata:
                                        st.markdown("**Metadata:**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown(f"- File: `{metadata.get('file_name', 'Unknown')}`")
                                            st.markdown(f"- Chunk Index: `{metadata.get('chunk_index', 'N/A')}`")
                                        with col2:
                                            st.markdown(f"- Length: `{metadata.get('original_length', 'N/A')}`")
                                    
                                    # Display text preview
                                    if 'text_preview' in metadata:
                                        st.markdown("**Text Preview:**")
                                        st.markdown("""```text
{}```""".format(metadata['text_preview'].strip()))
                                    
                                    st.markdown("---")
    else:
        # Main landing page
        st.header("Welcome to the RAG Pipeline with Airflow")
        st.markdown("""
        This application lets you use a Retrieval-Augmented Generation (RAG) pipeline 
        to query PDF documents or Nvidia's quarterly reports.
        
        ### Configuration Options:
        
        1. **Select your data source:**
           - Upload your own PDF documents
           - Query Nvidia quarterly reports from the past 5 years
        
        2. **Choose your PDF parsing method:**
           - Basic Parser
           - Docling
           - Mistral OCR
        
        3. **Select RAG implementation:**
           - Naive approach (Manual embeddings)
           - Pinecone vector database
           - ChromaDB vector database
        
        4. **Choose chunking strategy:**
           - Fixed Size
           - Semantic
           - Recursive
        
        ### Get Started:
        Configure your preferences in the sidebar and click "Apply Configuration".
        """)
        
        # System architecture diagram (placeholder)
        st.subheader("System Architecture")
        st.markdown("""
        ```
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │  Streamlit  │───►│   FastAPI   │───►│   Airflow   │
        │   (UI)      │◄───│  (Backend)  │◄───│  (Pipeline) │
        └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                           ┌─────────────┐    ┌─────────────┐
                           │ LLM Service │    │ Vector DB   │
                           │             │    │             │
                           └─────────────┘    └─────────────┘
        ```
        """)