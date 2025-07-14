# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a tutorial/guide for building a multimodal RAG (Retrieval-Augmented Generation) AI system that processes PDF documents. The system extracts text and images from PDFs, creates vector embeddings, and provides an AI-powered question-answering interface.

## Architecture Structure

The tutorial describes a modular architecture with clear separation of concerns:

```
ai-doc-assistant/
├── ingest/          # PDF text and image extraction
├── process/         # Text chunking and embedding generation  
├── index/           # Vector store management
├── backend/         # QA chain and API server
└── interface/       # User interface
```

Data flow: Ingest → Process → Index → Answer → Display

## Technology Stack

- **PDF Processing**: PyMuPDF (fitz) for text/image extraction
- **Text Embeddings**: sentence-transformers (all-MiniLM-L6-v2 model)
- **Image Embeddings**: CLIP (openai/clip-vit-base-patch32)
- **Vector Database**: FAISS for similarity search
- **LLM Framework**: LangChain with Ollama (Mistral model)
- **API Framework**: FastAPI
- **UI Framework**: Gradio
- **Database**: SQLite for interaction history
- **Deployment**: Docker containerization

## Key Implementation Details

- **Text Chunking**: Uses 500-character chunks with 100-character overlap to preserve context
- **Multimodal Search**: Supports both text and image-based queries using CLIP embeddings
- **Source Citation**: All answers include document references with file names and page numbers
- **Live Ingestion**: Supports real-time PDF uploads and index updates
- **Memory**: SQLite database tracks all interactions for continuous improvement

## Dependencies (Expected)

Based on the tutorial code, the system would require:
```
pymupdf
sentence-transformers
faiss-cpu
langchain
ollama
gradio
fastapi
uvicorn
torch
transformers
pillow
sqlite3
```

## Development Commands

Since this is currently a tutorial document, actual build/test commands would depend on implementation:

- **Install Dependencies**: `pip install -r requirements.txt`
- **Run API Server**: `uvicorn backend.server:app --host 0.0.0.0 --port 8000`
- **Launch UI**: `python interface/ui.py` (Gradio interface)
- **Docker Build**: `docker build -t ai-doc-assistant .`
- **Docker Run**: `docker run -p 8000:8000 ai-doc-assistant`

## Current Status

The repository currently contains only the tutorial document (`readpdf_Code.txt`). To implement the system, you would need to:

1. Create the directory structure as outlined
2. Implement each module based on the provided code snippets
3. Set up proper dependency management
4. Add configuration files and documentation