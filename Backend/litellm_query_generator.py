import os
import re
from typing import List, Dict, Any, Optional, Literal
import litellm
import time
from dotenv import load_dotenv
from pathlib import Path
# Try to find and load the .env file
def load_env_file():
    """Attempt to load environment variables from .env file in various locations"""
    # Current directory
    if os.path.exists(".env"):
        load_dotenv(".env")
        print("Loaded .env from current directory")
        return True
        
    # Project root directory (parent of current directory)
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from project root: {env_path}")
        return True
        
    # Airflow path (for when running in Airflow)
    airflow_env = "/opt/airflow/.env"
    if os.path.exists(airflow_env):
        load_dotenv(airflow_env, override=True)
        print(f"Loaded .env from Airflow path: {airflow_env}")
        return True
        
    print("WARNING: No .env file found!")
    return False

# Load environment variables
load_env_file()

# Function to force reload environment variables
def reload_env_variables():
    """Force reload environment variables from .env file, clearing any cached values"""
    # Clear specific environment variables we're using
    for key in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "DEEP_SEEK_API_KEY", 
                "OPENAI_API_KEY", "GROK_API_KEY"]:
        if key in os.environ:
            del os.environ[key]

    # Reload from .env file
    load_env_file()

# Configure LiteLLM with API keys
litellm.gemini_key = os.getenv("GEMINI_API_KEY")
litellm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
litellm.deepseek_api_key = os.getenv("DEEP_SEEK_API_KEY")
litellm.openai_api_key = os.getenv("OPENAI_API_KEY")
litellm.xai_api_key = os.getenv("GROK_API_KEY")

# Debug output to check API keys
print(f"API Keys Status:")
print(f"- Gemini:    {'✓' if litellm.gemini_key else '✗'}")
print(f"- Anthropic: {'✓' if litellm.anthropic_api_key else '✗'}")
print(f"- DeepSeek:  {'✓' if litellm.deepseek_api_key else '✗'}")
print(f"- OpenAI:    {'✓' if litellm.openai_api_key else '✗'}")
print(f"- Grok:      {'✓' if litellm.xai_api_key else '✗'}")

# Model configurations
MODEL_CONFIGS = {
    "gpt-3.5-turbo": {
        "name": "gpt-3.5-turbo",
        "model": "openai/gpt-3.5-turbo",
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
    },
    "gemini": {
        "name": "Gemini Flash",
        "model": "gemini/gemini-1.5-flash",
        "max_input_tokens": 100000,
        "max_output_tokens": 4000,
    },
    "deepseek": {
        "name": "DeepSeek",
        "model": "deepseek/deepseek-reasoner", 
        "max_input_tokens": 16000,
        "max_output_tokens": 2048,
    },
    "claude": {
        "name": "Claude 3 Sonnet",
        "model": "anthropic/claude-3-5-sonnet-20240620",
        "max_input_tokens": 100000,
        "max_output_tokens": 4096,
    },
    "grok": {
        "name": "Grok",
        "model": "xai/grok-2-latest", 
        "max_input_tokens": 8192,
        "max_output_tokens": 2048,
    }
}

def create_llm_response_from_chunks(
    chunks: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    query: str = "",
    model_id: str = "gpt-3.5-turbo"
) -> Dict[str, Any]:
    """
    Generate a Q&A response from text chunks using the specified model.
    
    Args:
        chunks: List of text chunks to process
        metadata: Optional list of metadata for each chunk
        query: The user's question
        model_id: ID of the model to use (from MODEL_CONFIGS)
        
    Returns:
        A dictionary containing the response and token usage information
    """
    try:
        # Validate inputs
        if not chunks:
            return {"content": "Error: No chunks provided for processing.", "usage": {"total_tokens": 0}}
            
        if not query:
            return {"content": "Error: Question must be provided for Q&A.", "usage": {"total_tokens": 0}}
            
        if model_id not in MODEL_CONFIGS:
            return {"content": f"Error: Unsupported model '{model_id}'. Choose from: {', '.join(MODEL_CONFIGS.keys())}", "usage": {"total_tokens": 0}}
        
        model_config = MODEL_CONFIGS[model_id]
        
        # Check API key availability for the selected model
        api_key_status = False
        
        if model_id == "gemini" and litellm.gemini_key:
            api_key_status = True
        elif model_id == "gpt-3.5-turbo" and litellm.openai_api_key:
            api_key_status = True
        elif model_id == "claude" and litellm.anthropic_api_key: 
            api_key_status = True
        elif model_id == "deepseek" and litellm.deepseek_api_key:
            api_key_status = True
        elif model_id == "grok" and litellm.xai_api_key:
            api_key_status = True
            
        if not api_key_status:
            return {"content": f"Error: API key not found for {model_id}. Please check your .env file.", "usage": {"total_tokens": 0}}
        
        # Prepare the context from chunks
        context = ""
        for i, chunk in enumerate(chunks):
            # Add metadata if available
            if metadata and i < len(metadata):
                meta = metadata[i]
                source = meta.get("source", "Unknown")
                similarity = meta.get("similarity_score", "N/A")
                context += f"\n## Source {i+1}: {source} (Similarity: {similarity})\n\n{chunk}\n\n"
            else:
                context += f"\n## Chunk {i+1}:\n\n{chunk}\n\n"
        
        # Create system message with instructions
        system_message = f"""You are a helpful AI assistant that answers questions based on the provided content.
        
Use the following context to answer the user's question:

{context}

When answering:
1. Only use information from the provided context
2. If the information isn't in the context, say "I don't have enough information to answer this question"
3. Be concise and focused in your response"""
        
        # Prepare the messages with both system and user content
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        # Call model using LiteLLM with specific configurations per model
        if model_id == "claude":
            response = litellm.completion(
                model=model_config["model"],  
                messages=messages,
                temperature=0.3,
                max_tokens=model_config["max_output_tokens"],
                api_key=litellm.anthropic_api_key
            )
        elif model_id == "gemini":
            response = litellm.completion(
                model=model_config["model"], 
                messages=messages,
                temperature=0.3,
                max_tokens=model_config["max_output_tokens"],
                api_key=litellm.gemini_key
            )
        elif model_id == "deepseek":
            response = litellm.completion(
                model=model_config["model"], 
                messages=messages,
                temperature=0.3,
                max_tokens=model_config["max_output_tokens"],
                api_key=litellm.deepseek_api_key
            )
        elif model_id == "grok":
            response = litellm.completion(
                model=model_config["model"],  
                messages=messages,
                temperature=0.3,
                max_tokens=model_config["max_output_tokens"],
                api_key=litellm.xai_api_key
            )
        else:  # Default to GPT-3.5-turbo
            response = litellm.completion(
                model=model_config["model"],
                messages=messages,
                temperature=0.3,
                max_tokens=model_config["max_output_tokens"],
                api_key=litellm.openai_api_key
            )
        
        # Extract and return the response with token usage
        if response and response.choices and response.choices[0].message.content:
            # Extract token usage information
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else 0,
                "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else 0
            }
            
            return {
                "content": response.choices[0].message.content,
                "usage": usage_info
            }
        else:
            return {
                "content": f"Error: No response generated from {model_config['name']}.",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    except Exception as e:
        return {
            "content": f"Error generating response: {str(e)}",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
def generate_response(
    chunks: List[str],
    query: str,
    model_id: str = "gpt-3.5-turbo",
    metadata: Optional[List[Dict[str, Any]]] = None,
    output_file: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate a response to a query based on the provided chunks.
    
    Args:
        chunks: List of text chunks from the RAG pipeline
        query: User's question
        model_id: ID of the model to use
        metadata: Optional metadata for the chunks
        output_file: Optional path to save the response
        images: Optional list of base64 encoded images from the chunks
        
    Returns:
        Dictionary with the generated answer and token usage information
    """
    print(f"Generating response using {MODEL_CONFIGS[model_id]['name']}")
    print(f"Query: {query}")
    print(f"Number of chunks: {len(chunks)}")
    if images:
        print(f"Number of images: {len(images)}")
    
    # Check if we should use a multimodal model with images
    has_images = images and len(images) > 0
    can_process_images = model_id in ["gemini", "claude", "gpt-4o"]
    
    if has_images and not can_process_images:
        print(f"Warning: Images found but model {model_id} cannot process images. Consider using gemini, claude, or gpt-4o")
    
    # For models that support images (like Gemini, Claude, GPT-4o)
    if has_images and can_process_images:
        # Generate response with images using multimodal capabilities
        response = create_llm_response_from_chunks(
            chunks=chunks,
            metadata=metadata,
            query=query,
            model_id=model_id,
            images=images
        )
    else:
        # Standard text-only response
        response = create_llm_response_from_chunks(
            chunks=chunks,
            metadata=metadata,
            query=query,
            model_id=model_id
        )
    
    answer = response["content"]
    usage = response["usage"]
    
    # Save the answer to a file if requested
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Sanitize query for filename if no output file specified
        if not os.path.basename(output_file):
            query_part = re.sub(r'[^\w\s-]', '', query)[:30].strip().replace(' ', '_')
            output_file = os.path.join(output_file, f"{model_id}_response_{query_part}.md")
            
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Query\n\n{query}\n\n# Response\n\n{answer}")
        
        print(f"Response saved to: {output_file}")
    
    return {
        "answer": answer, 
        "query": query, 
        "usage": usage,
        "model": model_id
    }




if __name__ == "__main__":
    # Test with some example chunks
    test_chunks = [
        "NVIDIA reported a revenue of $13.51 billion for Q1 2023, with a gross margin of 66.8%. The Data Center segment contributed $4.28 billion.",
        "NVIDIA's gaming revenue was $2.24 billion, down 38% from the previous year but up 22% from the previous quarter.",
        "NVIDIA's research and development expenses were $1.8 billion for the quarter, an increase of 40% from the previous year."
    ]
    
    test_metadata = [
        {"source": "NVIDIA Q1 2023 Report", "similarity_score": 0.92, "year": "2023", "quarter": "Q1"},
        {"source": "NVIDIA Q1 2023 Report", "similarity_score": 0.85, "year": "2023", "quarter": "Q1"},
        {"source": "NVIDIA Q1 2023 Report", "similarity_score": 0.78, "year": "2023", "quarter": "Q1"}
    ]
    
    test_query = "How much did the data center segment contribute to NVIDIA's revenue in Q1 2023?"
    
    # Test with default model (gpt-3.5-turbo)
    response = generate_response(
        chunks=test_chunks,
        query=test_query,
        metadata=test_metadata,
        model_id="gpt-3.5-turbo"
    )
    
    print("\nResponse Preview:")
    print(response["answer"][:500] + "..." if len(response["answer"]) > 500 else response["answer"])
    print(f"\nToken Usage: {response['usage']}")