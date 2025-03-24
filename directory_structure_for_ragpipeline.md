# Directory Structure Overview for RAG Pipeline

Here's a detailed breakdown of the directories and files used across the different components:

## 1. Mistral PDF Parser (mistralparsing_userpdf.py)

**Directories:**
- `pdfs_to_process/`: Where users place PDFs to be processed
- `pdfs-done/`: Where processed PDFs are moved after successful processing
- `ocr_output/`: Root folder for conversion results

**Files:**
- `ocr_output/{pdf_name}/output.md`: Markdown file with extracted text and embedded images

## 2. Docling Parser (doclingparsing.py)

**Directories:**
- `temp/`: Temporary storage during parsing
- `parsed_documents/`: Output directory for parsed documents

**Files:**
- `parsed_documents/{document_name}.md`: Cleaned markdown output

## 3. API (api.py)

**Directories:**
- uploads: Stores uploaded PDF files
- `markdown/`: Stores markdown conversions of PDFs
- `embeddings/`: Stores vector embeddings (if not using Pinecone)
- `jobs/`: Stores job status information

**Files:**
- `uploads/{file_id}.pdf`: Uploaded PDF files
- `markdown/{file_id}.md`: Markdown conversions
- `jobs/{job_id}.json`: Job status information

## 4. Pinecone RAG (rag_pinecone.py)

**Directories:**
- `chunk_storage/`: Stores JSON files with full chunk content
- `logs/`: Stores log files for the RAG pipeline
- `test_data/`: Sample test documents for testing the pipeline

**Files:**
- `chunk_storage/{index_name}_chunks.json`: JSON files with full content of chunks
- `logs/rag_pipeline.log`: Log file for the RAG pipeline
- `test_data/*.md`: Sample test documents

## Required Setup Before Running

1. **Environment Variables:**
   - Ensure you have a .env file with the following keys:
     - `OPENAI_API_KEY`: For OpenAI embeddings and completions
     - `PINECONE_API_KEY`: For Pinecone vector database
     - `MISTRAL_API_KEY`: For Mistral PDF parsing
     - Other optional API keys (GEMINI, ANTHROPIC, etc.)

2. **Create Necessary Directories:**
   - The code will create most directories automatically, but ensure you have write permissions

3. **Pinecone Connection:**
   - Ensure your Pinecone API key is valid
   - If using a specific index, note the index name for reference

4. **Test Documents:**
   - Have some test markdown files ready in the root directory (`Q1.md`, etc.)
   - Or prepare PDF files in the `pdfs_to_process/` directory

When you run the `rag_pinecone.py` file directly, it will test the full pipeline including loading documents to Pinecone and running queries against them. The interactive test will guide you through the entire process.