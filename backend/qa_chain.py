"""Simplified Question-answering chain using LangChain and Ollama."""

from typing import List, Dict, Any, Optional
from langchain.llms import Ollama
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#adjust ollama settings as well in .bashrc; verify w/ env | grep OLLAMA
class SimpleQAChain:
    """Simplified question-answering chain with document retrieval."""
    
    def __init__(self, model_name: str ="llama3.2:1b"): #"mistral"
        self.llm = Ollama(base_url="http://localhost:11434", 
                          model=model_name,
                          temperature=0.7,

                          #memory optimizations
                          num_ctx=1024,

                          #performance settings
                          repeat_penalty=1.1,
                          top_k=20,
                          top_p=0.9,
                          ) #adjust accordingly
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True}
                        )
        
        # Initialize the QA chain (this doesn't need a retriever)
        self.qa_chain = load_qa_chain(llm=self.llm, chain_type="stuff")
    
    def setup_chain(self, vector_store_obj):
        """
        Setup the QA chain with a VectorStore object.
        
        Args:
            vector_store_obj: Your custom VectorStore instance
        """
        try:
            print(f"Setting up simple QA chain with vector store: {type(vector_store_obj)}")
            
            # Get or create LangChain FAISS vector store from your custom VectorStore
            langchain_store = vector_store_obj.get_langchain_vector_store(self.embeddings)
            
            if langchain_store is None:
                print("❌ Failed to get LangChain vector store")
                print("Make sure you have added text embeddings to your vector store first.")
                return False
            
            # Store the LangChain vector store
            self.vector_store = langchain_store
            
            # Check if vector store has any documents
            if hasattr(langchain_store, 'index') and langchain_store.index.ntotal == 0:
                print("⚠️ Vector store is empty. Please add documents before running queries.")
                return False
            
            print("✅ Simple QA chain successfully initialized.")
            return True
            
        except Exception as e:
            import traceback
            print(f"❌ Failed to setup QA chain: {str(e)}")
            traceback.print_exc()
            return False
    
    def run_query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Run a query using manual document retrieval and QA chain.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Answer with source documents
        """
        if not self.vector_store:
            return {
                'answer': "Vector store not initialized. Please setup the chain first.",
                'source_documents': [],
                'question': question
            }
        
        if not self.qa_chain:
            return {
                'answer': "QA chain not initialized.",
                'source_documents': [],
                'question': question
            }
        
        try:
            # Step 1: Retrieve relevant documents
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
            
            if not docs_with_scores:
                return {
                    'answer': "No relevant documents found for your question.",
                    'source_documents': [],
                    'question': question
                }
            
            # Extract just the documents (without scores)
            docs = [doc for doc, score in docs_with_scores]
            
            # Step 2: Use QA chain to answer based on retrieved documents
            result = self.qa_chain({
                "input_documents": docs,
                "question": question
            })
            
            # Step 3: Format the response
            response = {
                'answer': result.get('output_text', 'No answer generated'),
                'source_documents': self._format_source_documents_with_scores(docs_with_scores),
                'question': question,
                'num_sources': len(docs)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error running query: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f"Error processing question: {str(e)}",
                'source_documents': [],
                'question': question
            }
    
    def _format_source_documents_with_scores(self, docs_with_scores: List[tuple]) -> List[Dict[str, Any]]:
        """Format source documents with similarity scores for display."""
        formatted_docs = []
        
        for i, (doc, score) in enumerate(docs_with_scores):
            formatted_doc = {
                'index': i + 1,
                'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score),
                'source': self._format_source_info(doc.metadata)
            }
            formatted_docs.append(formatted_doc)
        
        return formatted_docs
    
    def _format_source_info(self, metadata: Dict[str, Any]) -> str:
        """Format source information for display."""
        parts = []
        
        if 'filename' in metadata:
            parts.append(f"File: {metadata['filename']}")
        
        if 'page' in metadata:
            parts.append(f"Page: {metadata['page']}")
        
        if 'chunk_index' in metadata:
            parts.append(f"Section: {metadata['chunk_index']}")
        
        if 'source' in metadata:
            parts.append(f"Source: {metadata['source']}")
        
        return " | ".join(parts) if parts else "Unknown source"
    
    def get_relevant_documents(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant documents for a question without generating an answer.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        if not self.vector_store:
            return []
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
            return self._format_source_documents_with_scores(docs_with_scores)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about the QA chain."""
        stats = {
            'qa_chain_initialized': self.qa_chain is not None,
            'vector_store_loaded': self.vector_store is not None,
            'llm_model': getattr(self.llm, 'model', 'unknown'),
            'embedding_model': getattr(self.embeddings, 'model_name', 'unknown')
        }
        
        if self.vector_store and hasattr(self.vector_store, 'index'):
            stats['total_documents'] = self.vector_store.index.ntotal
        
        return stats