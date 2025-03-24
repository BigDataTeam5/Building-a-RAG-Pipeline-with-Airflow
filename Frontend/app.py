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

# FastAPI Base URL - Simple configuration
if "fastapi_url" not in st.session_state:
    config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secret.toml")
    if os.path.exists(config_path):
        config_data = toml.load(config_path)
        st.session_state.fastapi_url = config_data.get("connections", {}).get("FASTAPI_URL")
                
if "api_connected" not in st.session_state:
    st.session_state.api_connected = True

# Define function to update API endpoints based on the configured URL
def update_api_endpoints():
    base_url = st.session_state.fastapi_url
    
    # API Endpoints
    st.session_state.UPLOAD_PDF_API = f"{base_url}/upload-pdf"
    st.session_state.PARSE_PDF_API = f"{base_url}/parse-pdf"
    st.session_state.RAG_EMBEDDING_API = f"{base_url}/rag/create-embeddings"
    st.session_state.ASK_QUESTION_API = f"{base_url}/ask-question"
    st.session_state.GET_LLM_RESULT_API = f"{base_url}/get-llm-result"
    st.session_state.LLM_MODELS_API = f"{base_url}/llm/models"
    
    # RAG-specific endpoints
    st.session_state.NVIDIA_QUARTERS_API = f"{base_url}/nvidia/quarters"
    st.session_state.RAG_QUERY_API = f"{base_url}/rag/query"
    st.session_state.RAG_CONFIG_API = f"{base_url}/rag/config"


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
            response = requests.post(st.session_state.UPLOAD_PDF_API, files=files, params=params)

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

# Function to sanitize index name for Pinecone - Enhanced version
def sanitize_index_name(name):
    """
    Sanitize index name to match Pinecone requirements:
    - Lowercase alphanumeric characters or hyphens only
    - Replace spaces and invalid characters with hyphens
    - Make sure it's a valid name
    - Ensure the name is within allowed length
    """
    if not name or not isinstance(name, str):
        return "default-index"
    
    # Convert to lowercase
    sanitized = name.lower()
    
    # Replace all non-alphanumeric characters (except hyphens) with hyphens
    sanitized = re.sub(r'[^a-z0-9-]', '-', sanitized)
    
    # Replace multiple consecutive hyphens with a single hyphen
    sanitized = re.sub(r'-+', '-', sanitized)
    
    # Remove leading and trailing hyphens
    sanitized = sanitized.strip('-')
    
    # Ensure we have a valid name - if empty or only contained invalid chars
    if not sanitized or not re.match(r'^[a-z0-9]', sanitized):
        sanitized = "default-index"
    
    # Limit to reasonable length (Pinecone may have a limit)
    max_length = 45  # An assumed safe maximum
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        # Ensure we don't end with a hyphen
        sanitized = sanitized.rstrip('-')
    
    return sanitized

# Function to Submit RAG Query
def submit_rag_query(question, model, quarters=None):
    try:
        with st.spinner("⏳ Processing your question with RAG pipeline... This may take a moment."):
            # Prepare the request payload
            payload = {
                # REQUIRED fields by FastAPI:
                "query": question,
                "rag_method": st.session_state.rag_method,
                "model_id": model,
                
                # OPTIONAL fields:
                "data_source": st.session_state.data_source,
                "quarters": quarters,
                "similarity_metric": st.session_state.similarity_metric
            }
            
            # Special handling for manual_embedding
            if st.session_state.rag_method == "manual_embedding":
                payload["embedding_id"] = "direct_query"
            
            # Special handling for Pinecone
            if st.session_state.rag_method == "Pinecone":
                # Create a sanitized index name based on data source or file name
                index_base = "pinecone"  # Start with a simple prefix
                
                # Add data source info if available
                if st.session_state.data_source:
                    index_base += "-" + st.session_state.data_source.replace(" ", "")
                
                # Add quarter info if available (limited to first quarter to avoid long names)
                if quarters and len(quarters) > 0:
                    index_base += "-" + quarters[0]
                
                # Sanitize the index name
                index_name = sanitize_index_name(index_base)
                
                # Add debugging info visible in the UI
                st.info(f"Using Pinecone index name: '{index_name}'")
                
                payload["index_name"] = index_name
            
            # Now post as JSON
            response = requests.post(
                st.session_state.RAG_QUERY_API,
                json=payload
            )

            if response.status_code == 200:
                return response.json()
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
            return ["gpt-4"]  # Default fallback model
    except Exception as e:
        st.warning(f"Error fetching models: {str(e)}")
        return ["gpt-4"]  # Default fallback model

# Improved polling function with progress bar and timeout
def poll_for_llm_result(job_id, max_retries=15, interval=2):
    """Poll for LLM result with a progress bar and better timeout handling"""
    retries = 0
    
    # Create a progress bar
    progress_text = "Waiting for LLM to process your request..."
    progress_bar = st.progress(0)
    
    while retries < max_retries:
        try:
            # Calculate progress percentage
            progress = min(retries / max_retries, 0.95)  # Cap at 95% until complete
            progress_bar.progress(progress)
            
            # Check result status
            response = requests.get(
                f"{st.session_state.GET_LLM_RESULT_API}/{job_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                result_data = response.json()
                status = result_data.get("status")
                
                if status == "completed":
                    progress_bar.progress(1.0)  # Complete the progress bar
                    time.sleep(0.5)  # Brief pause to show completed progress
                    progress_bar.empty()  # Remove the progress bar
                    return result_data
                    
                elif status == "failed":
                    progress_bar.empty()
                    st.error(f"LLM processing failed: {result_data.get('error', 'Unknown error')}")
                    return None
                    
            retries += 1
            time.sleep(interval)
            
        except Exception as e:
            progress_bar.empty()
            st.error(f"Error while polling for result: {str(e)}")
            return None
    
    # If we get here, we've exceeded max retries
    progress_bar.empty()
    st.error(f"Timed out waiting for LLM response after {max_retries * interval} seconds.")
    return None
# Add this toke usage calculation function
def calculate_token_cost(model_id, usage_data):
    """Calculate cost based on model and token usage"""
    # Default rates (can be adjusted based on actual pricing)
    rates = {
        "gpt-4": {"input": 0.00003, "output": 0.00006},
        "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
        "claude-3": {"input": 0.000025, "output": 0.000075},
        "gemini-pro": {"input": 0.000001, "output": 0.000002},
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
def process_rag_embeddings(file_id, markdown_path, rag_method, chunking_strategy):
    try:
        with st.spinner("⏳ Creating embeddings... This may take a moment."):
            # Prepare the request payload as JSON body (not query params)
            payload = {
                "markdown_path": markdown_path,
                "rag_method": rag_method,
                "chunking_strategy": chunking_strategy,
                "embedding_model": "text-embedding-ada-002"
            }
            
            # Submit to API with JSON body instead of query params
            response = requests.post(
                st.session_state.RAG_EMBEDDING_API, 
                json=payload  # Changed from params to json
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to create embeddings: {response.text}")
                return {"error": response.text}
    except Exception as e:
        st.error(f"Error in embedding creation: {str(e)}")
        return {"error": str(e)}

# Add this new function after the process_rag_embeddings function
def poll_for_embedding_status(job_id, max_retries=30, interval=2):
    """Poll for embedding job status with progress bar and timeout handling"""
    retries = 0
    
    # Create a progress bar
    progress_text = "Creating embeddings and storing vectors in database..."
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while retries < max_retries:
        try:
            # Calculate progress percentage (cap at 95% until complete)
            progress = min(retries / max_retries, 0.95)
            progress_bar.progress(progress)
            
            # Update status message
            if retries < max_retries * 0.3:
                status_text.text("⏳ Processing document chunks...")
            elif retries < max_retries * 0.6:
                status_text.text("⏳ Creating vector embeddings...")
            else:
                status_text.text("⏳ Storing vectors in database...")
            
            # Check job status
            response = requests.get(
                f"{st.session_state.RAG_EMBEDDING_API}/status/{job_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                result_data = response.json()
                status = result_data.get("status")
                
                if status == "completed":
                    progress_bar.progress(1.0)  # Complete the progress bar
                    status_text.text("✅ Embeddings created successfully!")
                    time.sleep(0.5)  # Brief pause to show completed progress
                    return result_data
                    
                elif status == "failed":
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Embedding creation failed: {result_data.get('error', 'Unknown error')}")
                    return None
            
            retries += 1
            time.sleep(interval)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error while polling for embedding status: {str(e)}")
            return None
    
    # If we get here, we've exceeded max retries
    progress_bar.empty()
    status_text.empty()
    st.error(f"Timed out waiting for embeddings after {max_retries * interval} seconds.")
    return None

# Sidebar UI
with st.sidebar:
    st.title("RAG Pipeline Configuration")
    
    # Dropdown for data source
    data_source = st.selectbox(
        "Select Data Source:",
        ["Select an option", "PDF Upload", "Nvidia Dataset"],
        index=0
    )
    
    # Only show these options for PDF Upload
    if data_source == "PDF Upload":
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
    elif data_source == "Nvidia Dataset":
        # Set default values for Nvidia Dataset
        st.session_state.pdf_parser = "Mistral"  
        st.session_state.rag_method = "ChromaDB"  
        st.session_state.chunking_strategy = "semantic_chunking"  
        st.session_state.llm_model = st.session_state.available_models[0] if st.session_state.available_models else "gpt-4"  # Default model
        
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
            if pdf_parser == "Select a parser":
                st.error("Please select a PDF parser")
            elif rag_method == "Select a method":
                st.error("Please select a RAG method")
            elif chunking_strategy == "Select a strategy":
                st.error("Please select a chunking strategy")
            elif llm_model == "Select Model":
                st.error("Please select an LLM model")
            else:
                st.success("Configuration applied successfully")
                st.session_state.next_clicked = True
                st.rerun()
        elif data_source == "Nvidia Dataset" and not selected_quarters:
            st.error("Please select at least one quarter")
        else:
            st.success("Configuration applied successfully")
            st.session_state.next_clicked = True
            st.rerun()
    st.markdown("---")
    st.subheader("Analytics")
    if st.button("📊 View Token Usage"):
        st.session_state.show_token_usage = True
        st.rerun()
    

# Main Page Logic
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
                    
                # Replace the existing embedding button code (around line 462)
                if st.button("Process Chunking & Create Embeddings"):
                    # Start the embedding process
                    embedding_response = process_rag_embeddings(
                        file_id, 
                        markdown_path,
                        st.session_state.rag_method,
                        st.session_state.chunking_strategy
                    )
                    
                    if "error" not in embedding_response:
                        # Get the job ID from the response
                        job_id = embedding_response.get("job_id")
                        if job_id:
                            # Poll for job completion
                            final_result = poll_for_embedding_status(job_id, max_retries=60)
                            
                            if final_result and final_result.get("status") == "completed":
                                st.success("✅ Embeddings created successfully!")
                                st.info("You can now ask questions about your document.")
                                
                                # Store the job_id for later reference
                                st.session_state.embedding_job_id = job_id
                            else:
                                st.error("Failed to create embeddings. Please try again.")
                        else:
                            st.error("No job ID returned from the embedding creation process.")
                    else:
                        st.error(f"Embedding failed: {embedding_response.get('error')}")    
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
        similarity_metric = st.selectbox(
            "Select Similarity Metric:",
            [
                "cosine_similarity",
                "euclidean_distance",
                "dot_product" 
            ],
            help="Choose how similarity between text chunks will be calculated"
        )
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
                result = submit_rag_query(user_question, st.session_state.llm_model, quarters)
                
                if result:
                    # Display the answer
                    st.session_state.question_result = result
                    
                    # Show the answer with citations
                    st.markdown("### Answer")
                    answer_text = result.get("answer") or result.get("response") or "No answer available"
                    st.markdown(answer_text)
                    
                    # Show source documents if available
                    if "sources" in result and result["sources"]:
                        with st.expander("Source Documents", expanded=False):
                            for idx, source in enumerate(result["sources"]):
                                st.markdown(f"#### Source {idx+1}")
                                st.markdown(f"**Document:** {source.get('document', 'Unknown')}")
                                st.markdown(f"**Text:** {source.get('text', 'No text available')}")
                                st.markdown("---")

# Token usage display
if st.session_state.show_token_usage:
    st.title("📊 Token Usage Analytics")
    if st.button("🔙 Go Back"):
        st.session_state.show_token_usage = False
        st.rerun()
    st.markdown("### Token Usage History")
    if "token_usage_records" not in st.session_state:
        st.session_state.token_usage_records = []
    if hasattr(st.session_state, "question_result") and st.session_state.question_result:
        usage_data = st.session_state.question_result.get("usage", {})
        if usage_data:
            st.session_state.token_usage_records.append(usage_data)
        # extract usage data if available 
        if "usage" in result:
            if isinstance(usage_data, str):
                try:
                    usage_data = json.loads(usage_data)
                except:
                    usage_data = {"prompt_tokens": 0, "completion_tokens": 0,"total_tokens": 0}
            # Calculate costs based on the model
            model_id = result.get("model_id", "gpt-4")
            cost_data = calculate_token_cost(model_id, usage_data)

            #Create  a timestamp if not present
            if "timestamp" not in result:
                result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            #Add to usage records if not already there 
            query_id = result.get("job_id", str(uuid.uuid4()))

            # Check if this exact query is already recorded
            if not any(record.get("job_id")  == query_id for record in st.session_state.token_usage_records):
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