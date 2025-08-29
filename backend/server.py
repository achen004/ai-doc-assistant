"""FastAPI server for the AI document assistant."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional
import os
import tempfile
import logging
import json
import sqlite3

from ingest.document_extractor import DocumentExtractor
from process.text_processor import TextProcessor
from process.image_processor import ImageProcessor
from index.vector_store import VectorStore
from backend.qa_chain import SimpleQAChain

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

# Global components (will be initialized in startup)
document_extractor = None
text_processor = None
image_processor = None
vector_store = None
qa_chain = None


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


def save_interaction(question: str, answer: Union[str, QuestionResponse, Dict[str, Any]]):
    """Save user interaction to SQLite database."""
    try:
        # Ensure answer is JSON serializable
        if isinstance(answer, QuestionResponse):
            answer = answer.dict()   # convert Pydantic -> dict
        elif isinstance(answer, dict):
            answer = answer
        else:
            answer = str(answer)

        answer_str = json.dumps(answer, default=str, ensure_ascii=False)

        conn = sqlite3.connect("data/history.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO log (question, answer) VALUES (?, ?)",
            (str(question), answer_str)
        )
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
    
    sources_text = "\n\nSources:\n" + "\n".join(refs)
    return f"{answer}{sources_text}"


@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global document_extractor, text_processor, image_processor, vector_store, qa_chain
    
    logger.info("AI Document Assistant server starting up...")
    
    try:
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/docs", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize components
        logger.info("Initializing components...")
        document_extractor = DocumentExtractor()
        text_processor = TextProcessor()
        image_processor = ImageProcessor()
        vector_store = VectorStore()
        qa_chain = SimpleQAChain("tinyllama")
        
        # Load existing index
        logger.info("Loading vector store index...")
        vector_store.load_index()
        
        # Setup QA chain
        logger.info("Setting up QA chain...")
        setup_success = qa_chain.setup_chain(vector_store)
        
        if setup_success:
            logger.info("✅ QA chain setup successful")
        else:
            logger.warning("⚠️ QA chain setup failed - questions may not work properly")
        
        logger.info("✅ Server startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        import traceback
        traceback.print_exc()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Document Assistant API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = vector_store.get_stats() if vector_store else {}
        qa_stats = qa_chain.get_chain_stats() if qa_chain else {}
        
        return {
            "status": "healthy",
            "index_stats": stats,
            "qa_chain_stats": qa_stats
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document file (PDF, DOC, DOCX).
    
    Args:
        file: Document file to process
        
    Returns:
        Processing results
    """
    if not all([document_extractor, text_processor, image_processor, vector_store]):
        raise HTTPException(status_code=503, detail="Server components not initialized")
    
    # Check supported formats
    supported_extensions = ['.pdf', '.doc', '.docx']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
        )
    
    try:
        # Save uploaded file
        save_path = f"./data/docs/{file.filename}"
        os.makedirs("./data/docs", exist_ok=True)
        
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text and images
        logger.info(f"Processing document: {file.filename} ({file_ext})")
        extraction_result = document_extractor.extract_text_and_images(save_path)
        
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
        
        # Reinitialize QA chain with updated vector store
        if qa_chain:
            qa_chain.setup_chain(vector_store)
        
        return {
            "message": f"Successfully processed {file.filename}",
            "text_chunks": len(text_chunks),
            "images": image_count,
            "pages": extraction_result['metadata']['pages']
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/ask")
async def ask(query: Query):
    """
    Ask a question using the QA chain.
    
    Args:
        query: Question query
        
    Returns:
        Answer from QA chain
    """
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA chain not initialized")
    
    try:
        logger.info(f"Processing question: {query.question}")
        
        # Run the query
        response = qa_chain.run_query(query.question)
        
        # Handle response format
        if isinstance(response, dict):
            answer = response.get('answer', 'No answer generated')
            # Save the full response
            save_interaction(query.question, response)
        else:
            answer = str(response)
            # Save the string response
            save_interaction(query.question, answer)
        
        logger.info(f"Question processed successfully, answer length: {len(answer)}")
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@app.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded documents using vector search.
    
    Args:
        request: Question request with search parameters
        
    Returns:
        Answer with source citations
    """
    if not all([text_processor, image_processor, vector_store, qa_chain]):
        raise HTTPException(status_code=503, detail="Server components not initialized")
    
    try:
        question = request.question
        search_type = request.search_type
        k = request.k
        
        logger.info(f"Processing question with vector search: {question}")
        
        # Get query embeddings
        text_results = []
        image_results = []
        
        if search_type in ["text", "both"]:
            try:
                text_embedding = text_processor.generate_embeddings([question])[0]
                text_results = vector_store.search_text(text_embedding, k)
            except Exception as e:
                logger.error(f"Text search error: {str(e)}")
        
        if search_type in ["image", "both"]:
            try:
                image_embedding = image_processor.generate_text_embedding(question)
                image_results = vector_store.search_images(image_embedding, k)
            except Exception as e:
                logger.error(f"Image search error: {str(e)}")
        
        # Combine results
        all_results = text_results + image_results
        
        # Sort by score (lower scores are better for L2 distance)
        all_results.sort(key=lambda x: x.get('score', float('inf')))
        
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
        save_interaction(question, response)
        
        logger.info(f"Question processed successfully with {len(final_results)} sources")
        
        return QuestionResponse(**response)
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = {}
        
        if vector_store:
            stats.update(vector_store.get_stats())
        
        if qa_chain:
            qa_stats = qa_chain.get_chain_stats()
            stats.update(qa_stats)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"error": str(e)}


@app.get("/history")
async def get_history(limit: int = 50):
    """Get chat history."""
    try:
        conn = sqlite3.connect("data/history.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT question, answer, timestamp 
            FROM log 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        history = []
        for row in cursor.fetchall():
            try:
                answer_data = json.loads(row[1])
            except:
                answer_data = row[1]
            
            history.append({
                "question": row[0],
                "answer": answer_data,
                "timestamp": row[2]
            })
        
        conn.close()
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return {"history": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")