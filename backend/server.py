"""FastAPI server for the AI document assistant."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import tempfile
import logging
import sqlite3

from ingest.pdf_extractor import PDFExtractor
from process.text_processor import TextProcessor
from process.image_processor import ImageProcessor
from index.vector_store import VectorStore
from backend.qa_chain import QAChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Document Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_extractor = PDFExtractor()
text_processor = TextProcessor()
image_processor = ImageProcessor()
vector_store = VectorStore()
qa_chain = QAChain()

# Load existing index
vector_store.load_index()
qa_chain.setup_chain(vector_store)


class Query(BaseModel):
    question: str


class QuestionRequest(BaseModel):
    question: str
    search_type: str = "text"  # "text", "image", or "both"
    k: int = 5


class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    question: str


def save_interaction(question: str, answer: str):
    """Save user interaction to SQLite database."""
    try:
        conn = sqlite3.connect("data/history.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT INTO log (question, answer) VALUES (?, ?)", (question, answer))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving interaction: {str(e)}")


def format_response(answer: str, docs: List[Dict[str, Any]]) -> str:
    """Format response with source citations."""
    if not docs:
        return answer
    
    refs = []
    for doc in docs:
        source = doc.get('source', {})
        filename = source.get('filename', source.get('file', 'Unknown'))
        page = source.get('page', 'Unknown')
        refs.append(f"{filename} (p{page})")
    
    sources_text = "\\n\\nSources:\\n" + "\\n".join(refs)
    return f"{answer}{sources_text}"


@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("AI Document Assistant server starting up...")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Document Assistant API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "index_stats": stats
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    
    Args:
        file: PDF file to process
        
    Returns:
        Processing results
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        save_path = f"./data/docs/{file.filename}"
        os.makedirs("./data/docs", exist_ok=True)
        
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text and images
        logger.info(f"Processing PDF: {file.filename}")
        extraction_result = pdf_extractor.extract_text_and_images(save_path)
        
        # Process text pages
        text_chunks = []
        for page_data in extraction_result['text_pages']:
            if page_data['text'].strip():
                # Chunk the text
                chunks = text_processor.chunk_text(page_data['text'])
                
                # Add metadata to chunks
                for i, chunk in enumerate(chunks):
                    text_chunks.append({
                        'text': chunk,
                        'source': {
                            'filename': file.filename,
                            'file': page_data['file'],
                            'page': page_data['page'],
                            'chunk_index': i,
                            'type': 'text'
                        }
                    })
        
        # Generate text embeddings
        if text_chunks:
            texts = [chunk['text'] for chunk in text_chunks]
            embeddings = text_processor.embed_chunks(texts)
            vector_store.add_text_embeddings(embeddings, text_chunks)
        
        # Process images
        image_count = 0
        for img_data in extraction_result['image_metadata']:
            try:
                # Generate image embedding
                embedding = image_processor.embed_image(img_data['image_path'])
                
                # Add to vector store
                vector_store.add_image_embeddings([embedding], [{
                    'source': {
                        'filename': file.filename,
                        'file': img_data['file'],
                        'page': img_data['page'],
                        'image_path': img_data['image_path'],
                        'type': 'image'
                    }
                }])
                image_count += 1
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                continue
        
        # Save updated index
        vector_store.save_index()
        
        return {
            "message": f"Successfully processed {file.filename}",
            "text_chunks": len(text_chunks),
            "images": image_count,
            "pages": extraction_result['metadata']['pages']
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask")
async def ask(query: Query):
    """
    Ask a question using the QA chain.
    
    Args:
        query: Question query
        
    Returns:
        Answer from QA chain
    """
    try:
        response = qa_chain.run_query(query.question)
        
        # Save interaction
        save_interaction(query.question, response)
        
        return {"answer": response}
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@app.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded documents.
    
    Args:
        request: Question request with search parameters
        
    Returns:
        Answer with source citations
    """
    try:
        question = request.question
        search_type = request.search_type
        k = request.k
        
        # Get query embeddings
        if search_type in ["text", "both"]:
            text_embedding = text_processor.generate_embeddings([question])[0]
            text_results = vector_store.search_text(text_embedding, k)
        else:
            text_results = []
        
        if search_type in ["image", "both"]:
            image_embedding = image_processor.generate_text_embedding(question)
            image_results = vector_store.search_images(image_embedding, k)
        else:
            image_results = []
        
        # Combine results
        all_results = text_results + image_results
        
        # Sort by score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Limit results
        final_results = all_results[:k]
        
        # Generate answer using retrieved documents
        if final_results:
            response = qa_chain.answer_question(question, final_results)
        else:
            response = {
                'answer': 'No relevant documents found for your question.',
                'sources': [],
                'question': question
            }
        
        # Save interaction
        save_interaction(question, response['answer'])
        
        return QuestionResponse(**response)
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return vector_store.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)