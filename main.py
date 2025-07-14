"""
Main entry point for the AI Document Assistant.

This script demonstrates the complete pipeline from the readpdf_Code.txt tutorial.
"""

import os
import logging
from typing import List, Dict, Any

# Import our modules
from ingest.pdf_extractor import PDFExtractor
from process.text_processor import TextProcessor
from process.image_processor import ImageProcessor
from index.vector_store import VectorStore
from backend.qa_chain import QAChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main demonstration of the AI Document Assistant pipeline.
    """
    print("🤖 AI Document Assistant - Main Pipeline Demo")
    print("=" * 50)
    
    # Initialize components
    print("Initializing components...")
    pdf_extractor = PDFExtractor()
    text_processor = TextProcessor()
    image_processor = ImageProcessor()
    vector_store = VectorStore()
    qa_chain = QAChain()
    
    # Create data directories
    os.makedirs("data/docs", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    
    print("✅ Components initialized successfully!")
    
    # Example usage with a sample PDF (if available)
    sample_pdf = "data/docs/sample.pdf"
    
    if os.path.exists(sample_pdf):
        print(f"\\n📄 Processing sample PDF: {sample_pdf}")
        
        # Step 1: Extract text and images
        print("1. Extracting text and images...")
        extraction_result = pdf_extractor.extract_text_and_images(sample_pdf)
        
        # Step 2: Process text chunks
        print("2. Processing text chunks...")
        text_chunks = []
        for page_data in extraction_result['text_pages']:
            if page_data['text'].strip():
                chunks = text_processor.chunk_text(page_data['text'])
                for i, chunk in enumerate(chunks):
                    text_chunks.append({
                        'text': chunk,
                        'source': {
                            'file': page_data['file'],
                            'page': page_data['page'],
                            'chunk_index': i
                        }
                    })
        
        # Step 3: Generate embeddings
        print("3. Generating embeddings...")
        if text_chunks:
            texts = [chunk['text'] for chunk in text_chunks]
            embeddings = text_processor.embed_chunks(texts)
            vector_store.add_text_embeddings(embeddings, text_chunks)
        
        # Step 4: Process images
        print("4. Processing images...")
        image_count = 0
        for img_data in extraction_result['image_metadata']:
            try:
                embedding = image_processor.embed_image(img_data['image_path'])
                vector_store.add_image_embeddings([embedding], [img_data])
                image_count += 1
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
        
        # Step 5: Save index
        print("5. Saving vector index...")
        vector_store.save_index()
        
        # Step 6: Setup QA chain
        print("6. Setting up QA chain...")
        qa_chain.setup_chain(vector_store)
        
        print(f"\\n✅ Processing complete!")
        print(f"   📝 Text chunks: {len(text_chunks)}")
        print(f"   🖼️ Images: {image_count}")
        print(f"   📊 Total embeddings: {vector_store.get_stats()['total_embeddings']}")
        
        # Example question
        print("\\n❓ Example Question:")
        question = "What is this document about?"
        answer = qa_chain.run_query(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        
    else:
        print(f"\\n📂 No sample PDF found at {sample_pdf}")
        print("   To test the system:")
        print("   1. Place a PDF file at data/docs/sample.pdf")
        print("   2. Run this script again")
        print("   3. Or use the web interface: python interface/ui.py")
    
    print("\\n🚀 To start the web interface:")
    print("   Backend: uvicorn backend.server:app --host 0.0.0.0 --port 8000")
    print("   Frontend: python interface/ui.py")
    
    print("\\n🐳 Or use Docker:")
    print("   docker-compose up --build")


if __name__ == "__main__":
    main()