# AI Document Assistant

A multimodal RAG (Retrieval-Augmented Generation) AI system that processes PDF documents, extracts text and images, creates vector embeddings, and provides an AI-powered question-answering interface.

## Features

- **Document Processing**: Extract text and images from PDF and Word documents (PDF, DOC, DOCX)
- **Multimodal Embeddings**: Support for both text and image-based queries using CLIP
- **Vector Search**: FAISS-powered similarity search for efficient retrieval
- **AI Question Answering**: LangChain integration with Ollama
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
- **Vector Database**: LangChain's FAISS
- **LLM Framework**: LangChain with Ollama
- **API Framework**: FastAPI
- **UI Framework**: Gradio
- **Database**: SQLite
- **Deployment**: Docker

## Getting Started

### Prerequisites

- Python 3.9+
- Ollama installed with Mistral model (tinyllama if lower GPU settings)

set in ~/.bashrc
export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_GPU=8
export OLLAMA_NUM_THREAD=4

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
-tinyllama
-phi3:mini "phi3:mini-4k-instruct-q4_0"
-gemma:2b "gemma:2b-instruct-q4_0"
-qwen2:1.5b "codeqwen:1.5b-chat-q4_0"
-llama3.2:1b
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

- **Text Chunking**: 300-character chunks with 25-character overlap [text_processor.py]
- **Multimodal Search**: Supports both text and image queries
- **Memory**: All interactions stored for continuous improvement

## License

MIT License
