# Production RAG (Retrieval-Augmented Generation) System

A modular, production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, Google Generative AI (Gemini), Pinecone, and Streamlit.

This project allows you to ingest PDF documents, clean and chunk their contents, store vector embeddings in a Pinecone vector database, and perform context-aware Q&A with source citations.

---

## 🛠️ Features

- **PDF Ingestion & Processing**: Extracts text using PyMuPDF and splits it into manageable chunks with LangChain.
- **Dynamic Model Fallbacks**: Automatically handles Google embedding (`text-embedding-004`) and chat (`gemini-1.5-flash`, `gemini-2.0-flash`, etc.) models.
- **Vector Search**: Embeds document chunks and manages index creation and upserting in Pinecone.
- **REST API**: Built with FastAPI for quick integration with external services.
- **Web UI**: Interactive Streamlit dashboard to upload PDFs and chat with indexed documents.

---

## 📁 Project Structure

..

├── api.py                  # FastAPI application exposing /health, /ingest, and /ask

├── config.py               # Environment configuration and validation loader

├── embedding_provider.py   # Embedding provider with model fallback support

├── ingest.py               # PDF text extraction, chunking, and Pinecone upsert logic

├── llm_provider.py         # Google Gemini LLM provider with model fallback support

├── make_sample_pdf.py      # Utility script to generate a sample test PDF

├── query.py                # Retrieval and LLM QA generation pipeline

├── requirements.txt        # Project dependencies

└── streamlit_app.py        # Streamlit frontend user interface

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ installed on your machine.
- A Google Gemini API Key
- A Pinecone API Key

---

### Setup Instructions

1. Clone the repository:
   git clone https://github.com/prashantchauhan504/RAG.git
   cd RAG

2. Create and activate a virtual environment:

   - On macOS/Linux:
     python3 -m venv venv
     source venv/bin/activate

   - On Windows (Command Prompt):
     python -m venv venv
     venv\Scripts\activate

   - On Windows (PowerShell):
     python -m venv venv
     .\venv\Scripts\Activate.ps1

3. Install dependencies:
   pip install -r requirements.txt

4. Create environment configuration:
   Create a `.env` file in the root directory and add your API keys:

   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here

   # Optional Settings (Defaults will be used if omitted)
   PINECONE_INDEX_NAME=rag-index
   CHUNK_SIZE=500
   CHUNK_OVERLAP=50
   TOP_K=5
   GOOGLE_LLM_MODEL=gemini-1.5-flash
   EMBEDDING_MODEL=models/text-embedding-004
   ALLOW_INDEX_RECREATE=true

---

## 💻 Commands to Run the Project

### Option A: Run the Web Application (FastAPI + Streamlit)

1. Start the FastAPI backend server:
   uvicorn api:app --reload

   - API Server URL: http://127.0.0.1:8000
   - Health Check Endpoint: http://127.0.0.1:8000/health
   - API Docs (Swagger UI): http://127.0.0.1:8000/docs

2. Start the Streamlit frontend UI (in a new terminal window):
   streamlit run streamlit_app.py

   - Web Interface URL: http://localhost:8501

---

### Option B: Run via Command Line Interface (CLI)

1. Generate a sample PDF for testing:
   python make_sample_pdf.py

2. Ingest a PDF file into Pinecone:
   python ingest.py sample_for_chunking.pdf
   # or
   python ingest.py path/to/your/document.pdf

3. Query the index directly from terminal:
   python query.py "What is the summary of the document?"

---

## 📡 API Endpoints Summary

- GET /health
  Description: Health check route to verify backend status.
  Command / Call: curl http://127.0.0.1:8000/health

- POST /ingest?pdf_path=...
  Description: Ingests, chunks, embeds, and indexes the specified PDF file.
  Command / Call: curl -X POST "http://127.0.0.1:8000/ingest?pdf_path=sample_for_chunking.pdf"

- POST /ask?question=...
  Description: Retrieves relevant context from Pinecone and answers using Gemini.
  Command / Call: curl -X POST "http://127.0.0.1:8000/ask?question=What%20is%20in%20this%20document?"

---

## 🛡️ Git Ignore Recommendations

Create a `.gitignore` file before pushing to GitHub to avoid exposing keys:

.env
venv/
__pycache__/
uploaded_pdfs/
sample_for_chunking.pdf
*.pyc

---
