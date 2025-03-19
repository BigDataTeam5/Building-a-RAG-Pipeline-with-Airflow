import streamlit as st
import requests
import time
import sys
import os
import json
import uuid
import toml
# Streamlit UI
st.set_page_config(page_title="Nvidia Quarterly data RAG", layout="wide")

# Initialize session state variables if they do not exist
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "markdown_ready" not in st.session_state:
    st.session_state.markdown_ready = False
if "show_pdf_uploader" not in st.session_state:
    st.session_state.show_pdf_uploader = False
if "show_url_input" not in st.session_state:
    st.session_state.show_url_input = False
# Initialize session state for markdown history
if "markdown_history" not in st.session_state:
    st.session_state.markdown_history = []  # To store history of markdown files
if "selected_markdown_content" not in st.session_state:
    st.session_state.selected_markdown_content = None
if "selected_markdown_name" not in st.session_state:
    st.session_state.selected_markdown_name = None
# Initialize session state for LLM responses
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "question_result" not in st.session_state:
    st.session_state.question_result = None
if "processing_summary" not in st.session_state:
    st.session_state.processing_summary = False
if "processing_question" not in st.session_state:
    st.session_state.processing_question = False

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
    st.session_state.LATEST_FILE_API = f"{base_url}/get-latest-file-url"
    st.session_state.PARSE_PDF_API = f"{base_url}/parse-pdf"
    st.session_state.CONVERT_MARKDOWN_API = f"{base_url}/convert-pdf-markdown"
    st.session_state.FETCH_MARKDOWN_API = f"{base_url}/fetch-latest-markdown-urls"
    st.session_state.FETCH_MARKDOWN_HISTORY = f"{base_url}/list-image-ref-markdowns"
    st.session_state.SUMMARIZE_API = f"{base_url}/summarize"
    st.session_state.ASK_QUESTION_API = f"{base_url}/ask-question"
    st.session_state.GET_LLM_RESULT_API = f"{base_url}/get-llm-result"
    st.session_state.LLM_MODELS_API = f"{base_url}/llm/models"
    
    # RAG-specific endpoints
    st.session_state.NVIDIA_QUARTERS_API = f"{base_url}/nvidia/quarters"
    st.session_state.RAG_QUERY_API = f"{base_url}/rag/query"
    st.session_state.RAG_CONFIG_API = f"{base_url}/rag/config"

# Initial setup of API endpoints
update_api_endpoints()

# Function to Upload File to S3 - With improved error handling
def upload_pdf(file):
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        with st.spinner("📤 Uploading PDF... Please wait."):
            # Include RAG configuration parameters
            params = {
                "parser": st.session_state.pdf_parser,
                "rag_method": st.session_state.rag_method,
                "chunking_strategy": st.session_state.chunking_strategy
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

# Function to Submit RAG Query
def submit_rag_query(question, model, quarters=None):
    try:
        with st.spinner("⏳ Processing your question with RAG pipeline... This may take a moment."):
            request_id = f"rag_{uuid.uuid4()}"
            
            # Prepare the request payload
            payload = {
                "request_id": request_id,
                "question": question,
                "model": model,
                "parser": st.session_state.pdf_parser,
                "rag_method": st.session_state.rag_method,
                "chunking_strategy": st.session_state.chunking_strategy,
                "quarters": quarters
            }
            
            # Submit to API
            response = requests.post(
                st.session_state.RAG_QUERY_API, 
                json=payload
            )
            
            if response.status_code == 202:
                # Got a job ID, need to poll for result
                job_id = response.json().get("request_id")
                result = poll_for_llm_result(job_id)
                
                if result and "error" not in result:
                    return result
                else:
                    st.error(f"Error getting answer: {result.get('error', 'Unknown error')}")
                    return None
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

# Fetch models at startup
if "available_models" not in st.session_state:
    st.session_state.available_models = fetch_available_models()

# Sidebar UI
with st.sidebar:
    st.title("RAG Pipeline Configuration")
    
    # Dropdown for data source
    data_source = st.selectbox(
        "Select Data Source:",
        ["Select an option", "PDF Upload", "Nvidia Dataset"],
        index=0
    )
    
    # PDF parser selection
    pdf_parser = st.selectbox(
        "Select PDF Parser:",
        ["Select a parser", "Basic Parser", "Docling", "Mistral OCR"],
        index=0
    )
    
    # RAG method selection
    rag_method = st.selectbox(
        "Select RAG Method:",
        ["Select a method", "Naive (Manual Embeddings)", "Pinecone", "ChromaDB"],
        index=0
    )
    
    # Chunking strategy
    chunking_strategy = st.selectbox(
        "Select Chunking Strategy:",
        ["Select a strategy", "Fixed Size", "Semantic", "Recursive"],
        index=0
    )
    
    # If Nvidia dataset is selected, show quarter selection
    if data_source == "Nvidia Dataset":
        available_quarters = fetch_nvidia_quarters()
        if available_quarters:
            selected_quarters = st.multiselect(
                "Select Quarters to Query:",
                available_quarters,
                default=available_quarters[:1] if available_quarters else []
            )
            st.session_state.selected_quarters = selected_quarters
    
    # LLM model selection
    available_models = ["Select Model"] + st.session_state.available_models
    llm_model = st.selectbox("Select LLM Model:", available_models, index=0)
    
    # Save selections to session state
    st.session_state.data_source = data_source
    st.session_state.pdf_parser = pdf_parser if pdf_parser != "Select a parser" else None
    st.session_state.rag_method = rag_method if rag_method != "Select a method" else None
    st.session_state.chunking_strategy = chunking_strategy if chunking_strategy != "Select a strategy" else None
    st.session_state.llm_model = llm_model if llm_model != "Select Model" else None

    # Apply configuration button
    if st.button("Apply Configuration"):
        if data_source == "Select an option":
            st.error("Please select a data source")
        elif pdf_parser == "Select a parser":
            st.error("Please select a PDF parser")
        elif rag_method == "Select a method":
            st.error("Please select a RAG method")
        elif chunking_strategy == "Select a strategy":
            st.error("Please select a chunking strategy")
        elif llm_model == "Select Model":
            st.error("Please select an LLM model")
        elif data_source == "Nvidia Dataset" and not selected_quarters:
            st.error("Please select at least one quarter")
        else:
            st.success("Configuration applied successfully")
            st.session_state.next_clicked = True
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
                    answer_text = result.get("answer", "No answer available")
                    st.markdown(answer_text)
                    
                    # Show source documents if available
                    if "sources" in result and result["sources"]:
                        with st.expander("Source Documents", expanded=False):
                            for idx, source in enumerate(result["sources"]):
                                st.markdown(f"#### Source {idx+1}")
                                st.markdown(f"**Document:** {source.get('document', 'Unknown')}")
                                st.markdown(f"**Text:** {source.get('text', 'No text available')}")
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