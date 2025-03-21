# Building-a-RAG-Pipeline-with-Airflow
We are enhancing the llm extractor project in this organization , to implement rag concepts for reducing the input tokens


📌 Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using Apache Airflow for orchestrating data pipelines and FastAPI + Streamlit for interactive user interfaces. The goal is to build a scalable, modular system that extracts insights from NVIDIA’s quarterly reports (past 5 years) using various parsing and retrieval techniques.

🧠 Key Features
Data Source: NVIDIA quarterly reports (last 5 years)
PDF Parsing:
Assignment 1's Parser
Docling
Mistral OCR
RAG Pipeline Options:
Manual Embeddings + Cosine Similarity
Pinecone Integration
ChromaDB Integration
Chunking Strategies:
Fixed-size chunks
Semantic chunks
Sliding window chunks
Hybrid Search: Query by quarter
Interactive UI:
Upload PDFs
Select parser, chunking strategy, and RAG method
Query quarter-specific data
Deployment:
Dockerized Airflow pipeline
Dockerized Streamlit + FastAPI interface
🚀 Getting Started
🔧 Requirements
Docker & Docker Compose
Python 3.9+
Conda (optional for local development)
NVIDIA reports (downloaded into /data/raw_reports)
📂 Project Structure
kotlin
Copy
Edit
.
├── airflow/
│   └── dags/
├── app/
│   ├── fastapi_backend/
│   └── streamlit_frontend/
├── data/
│   ├── raw_reports/
│   └── processed/
├── utils/
├── Dockerfile.airflow
├── Dockerfile.app
├── docker-compose.yml
├── requirements.txt
└── README.md
🧪 Running the Project
Option 1: Run with Docker
bash
Copy
Edit
# Start all containers
docker-compose up --build
Option 2: Manual Setup (Local Dev)
Create a virtual environment or conda environment
Install requirements:
bash
Copy
Edit
pip install -r requirements.txt
Run Airflow scheduler & webserver
Run Streamlit & FastAPI locally
🧰 Technologies Used
Apache Airflow - Workflow orchestration
Streamlit - Interactive web UI
FastAPI - API backend
Pinecone / ChromaDB - Vector DBs for retrieval
Docling, Mistral OCR - PDF parsing
Docker - Containerization
🧪 Testing
Unit tests for data extraction and embedding
End-to-end tests for RAG pipeline
UI testing via demo walkthrough
📊 Project Tasks & Contribution
Member 1: Data ingestion + Airflow pipelines (33%)
Member 2: RAG system + Embedding strategies (33%)
Member 3: Streamlit/FastAPI UI + Deployment (33%)
Task breakdown and GitHub Issues: GitHub Project Board

📎 Submission Includes
✅ Project summary & Proof of Concept (PoC)
✅ CodeLab documentation
✅ Diagrams for architecture and pipeline
✅ 5-minute demo video
✅ Hosted URL of frontend/backend
🤖 AI Tools Disclosure
Documented in AIUseDisclosure.md

🏁 Evaluation Breakdown
Category	Weightage
Data Extraction & Parsing	25%
RAG Implementation & Chunking	40%
Streamlit UI & FastAPI	15%
Deployment & Dockerization	10%
Documentation & Presentation	10%
📚 References
Docling GitHub
Mistral OCR
Apache Airflow Docs
Pinecone Docs
ChromaDB Docs
Streamlit Docs
FastAPI Docs
