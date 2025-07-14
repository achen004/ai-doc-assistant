# AI Document Assistant

A multimodal RAG (Retrieval-Augmented Generation) AI system that processes PDF documents, extracts text and images, creates vector embeddings, and provides an AI-powered question-answering interface.

## Features

- **PDF Processing**: Extract text and images from PDF documents using PyMuPDF
- **Multimodal Embeddings**: Support for both text and image-based queries using CLIP
- **Vector Search**: FAISS-powered similarity search for efficient retrieval
- **AI Question Answering**: LangChain integration with Ollama (Mistral model)
- **Web Interface**: User-friendly Gradio interface
- **Source Citation**: All answers include document references with file names and page numbers
- **Live Ingestion**: Real-time PDF uploads and index updates
- **Interaction History**: SQLite database tracks all interactions

## Architecture

```
ai-doc-assistant/
├── ingest/          # PDF text and image extraction
├── process/         # Text chunking and embedding generation  
├── index/           # Vector store management
├── backend/         # QA chain and API server
└── interface/       # User interface
```

## Technology Stack

- **PDF Processing**: PyMuPDF (fitz)
- **Text Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Image Embeddings**: CLIP (openai/clip-vit-base-patch32)
- **Vector Database**: FAISS
- **LLM Framework**: LangChain with Ollama
- **API Framework**: FastAPI
- **UI Framework**: Gradio
- **Database**: SQLite
- **Deployment**: Docker

## Getting Started

### Prerequisites

- Python 3.8+
- Ollama installed with Mistral model

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-doc-assistant.git
cd ai-doc-assistant

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run API Server
uvicorn backend.server:app --host 0.0.0.0 --port 8000

# Launch UI
python interface/ui.py
```

### Docker Deployment

```bash
# Build image
docker build -t ai-doc-assistant .

# Run container
docker run -p 8000:8000 ai-doc-assistant
```

## Implementation Details

- **Text Chunking**: 500-character chunks with 100-character overlap
- **Multimodal Search**: Supports both text and image queries
- **Memory**: All interactions stored for continuous improvement

## Current Status

This repository contains the tutorial documentation and setup files. Implementation of the modular architecture is in progress.

## License

MIT License