# AI Document Assistant

A multimodal RAG (Retrieval-Augmented Generation) AI system that processes PDF documents, extracts text and images, creates vector embeddings, and provides an AI-powered question-answering interface.

## Features

- **Document Processing**: Extract text and images from PDF and Word documents (PDF, DOC, DOCX)
- **Multimodal Embeddings**: Support for both text and image-based queries using CLIP
- **Vector Search**: FAISS-powered similarity search for efficient retrieval
- **AI Question Answering**: LangChain integration with Ollama (Mistral model)
- **Web Interface**: User-friendly Streamlit interface (Windows-compatible)
- **Source Citation**: All answers include document references with file names and page numbers
- **Live Ingestion**: Real-time document uploads and index updates
- **Interaction History**: SQLite database tracks all interactions
- **Multi-format Support**: PDF, DOC, DOCX file processing

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

# Install and start Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull mistral
```

### Quick Start

```bash
# Option 1: Run everything with one command
python run.py

# Option 2: Run components separately
# Terminal 1: Start backend
uvicorn backend.server:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend  
python interface/ui.py

# Option 3: Run demo pipeline
python main.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build image manually
docker build -t ai-doc-assistant .
docker run -p 8000:8000 -p 7860:7860 ai-doc-assistant
```

## Implementation Details

- **Text Chunking**: 500-character chunks with 100-character overlap
- **Multimodal Search**: Supports both text and image queries
- **Memory**: All interactions stored for continuous improvement

## Current Status

This repository contains the tutorial documentation and setup files. Implementation of the modular architecture is in progress.

## License

MIT License