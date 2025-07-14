"""Question-answering chain using LangChain and Ollama."""

from typing import List, Dict, Any, Optional
from langchain.llms import Ollama
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAChain:
    """Question-answering chain with document retrieval."""
    
    def __init__(self, model_name: str = "mistral"):
        self.llm = Ollama(model=model_name)
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def setup_chain(self, vector_store):
        """
        Setup the QA chain with a vector store.
        
        Args:
            vector_store: Vector store instance for document retrieval
        """
        self.vector_store = vector_store
        
        # Create retriever from FAISS vector store
        if hasattr(vector_store, 'text_index') and vector_store.text_index.ntotal > 0:
            # Load FAISS index with LangChain wrapper
            retriever = FAISS.load_local("data/vector_index", embeddings=self.embeddings)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, 
                retriever=retriever.as_retriever()
            )
    
    def run_query(self, question: str) -> str:
        """
        Run a query using the QA chain.
        
        Args:
            question: User's question
            
        Returns:
            Answer from the QA chain
        """
        if self.qa_chain:
            return self.qa_chain.run(question)
        else:
            return "QA chain not initialized. Please add documents first."
    
    def answer_question(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Answer a question using retrieved documents.
        
        Args:
            question: User's question
            context_docs: Retrieved documents for context
            
        Returns:
            Answer with source citations
        """
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(context_docs)
            
            # Create prompt
            prompt = f"""
            Based on the following context, answer the question. If you cannot answer based on the context, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            # Get answer from LLM
            answer = self.llm(prompt)
            
            # Prepare response with sources
            response = {
                'answer': answer,
                'sources': self._extract_sources(context_docs),
                'question': question
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in QA chain: {str(e)}")
            return {
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'question': question
            }
    
    def _prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(docs):
            if 'text' in doc:
                source_info = self._format_source_info(doc.get('source', {}))
                context_parts.append(f"[Document {i+1}] {source_info}\n{doc['text']}")
        
        return "\n\n".join(context_parts)
    
    def _format_source_info(self, source: Dict[str, Any]) -> str:
        """Format source information for display."""
        parts = []
        
        if 'filename' in source:
            parts.append(f"File: {source['filename']}")
        
        if 'page' in source:
            parts.append(f"Page: {source['page']}")
        
        if 'chunk_index' in source:
            parts.append(f"Section: {source['chunk_index']}")
        
        return " | ".join(parts) if parts else "Unknown source"
    
    def _extract_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        sources = []
        
        for doc in docs:
            source = doc.get('source', {})
            sources.append({
                'filename': source.get('filename', 'Unknown'),
                'page': source.get('page', 'Unknown'),
                'score': doc.get('score', 0.0),
                'text_preview': doc.get('text', '')[:100] + "..." if 'text' in doc else ""
            })
        
        return sources