"""Simplified Question-answering chain using LangChain and Ollama."""

from typing import List, Dict, Any, Optional
from langchain.llms import Ollama
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#adjust ollama settings as well in .bashrc; verify w/ env | grep OLLAMA
class SimpleQAChain:
    """Simplified question-answering chain with document retrieval.
    Models to try:
    -"tinyllama"
    -ollama pull phi3:mini "phi3:mini-4k-instruct-q4_0"
    -ollama pull gemma:2b "gemma:2b-instruct-q4_0"
    -ollama pull qwen2:1.5b "codeqwen:1.5b-chat-q4_0"
    -"llama3.2:1b"
    -"mistral" 
    """
    
    def __init__(self, model_name: str ="tinyllama"): 
        self.llm = Ollama(base_url="http://localhost:11434", 
                          model=model_name,
                          temperature=0.1,

                          #memory optimizations
                          num_ctx=2048, #context window

                          #performance settings
                          repeat_penalty=1.05,
                          top_k=10,
                          top_p=0.9,
                          ) #adjust accordingly
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2", #bge-small-en-v1.5
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True}
                        )
        
        self.prompt_template=self._create_prompt_template()
        
        # Initialize the QA chain 
        self.qa_chain = load_qa_chain(llm=self.llm, 
                                      chain_type="stuff",
                                      prompt=self.prompt_template)
    
    #Custom prompt template
    def _create_prompt_template(self) -> PromptTemplate:
        """Custom prompt template for QA Chain"""
        template="""You are a helpful assistant that answers questions about the provided context.
        
        CONTEXT:
        {context}
        
        INSTRUCTIONS:
        -Answer the question using ONLY the information provided in the context
        -If the answer is explicitly stated in the context, provide it clearly and directly
        -If the answer requires combining information from multiple parts of the context, carefully proceed
        -If you cannot find the answer in the provided context, respond "Unable to find the specific answer to your question"
        -Be specific and cite relevant information from the context when possible
        -Keep your answer concise but complete
        
        QUESTION: {question}
        
        ANSWER:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def update_prompt_template(self, custom_template: str) -> bool:
        """
        Args: custom_template: custom prompt template string. Needs to include {context} and {question} placeholders

        Returns: True if successful, otherwise False
        """
        try:
            if "{context}" not in custom_template or "{question}" not in custom_template:
                logger.error("Template must include both {context} and {question} placeholders")
                return False
            #new prompt template
            new_prompt=PromptTemplate(
                template=custom_template,
                input_variables=["context","question"]
            )
            
            # Update the QA chain with new prompt
            self.prompt_template = new_prompt
            self.qa_chain = load_qa_chain(
                llm=self.llm, 
                chain_type="stuff",
                prompt=new_prompt
            )

            logger.info("Prompt template updated successfully")

            return True
        
        except Exception as e:
            logger.error(f"Failed to update prompt template:{str(e)}")
            return False

    
    def setup_chain(self, vector_store_obj):
        """
        Setup the QA chain with a VectorStore object. Ran in server.py
        
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

            #log retrieved context for debugging
            total_context_chars=sum(len(doc.page_content) for doc in docs)
            logger.info(f"Retrieved {len(docs)} documents, total context:  {total_context_chars} characters")
            
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
        if not metadata:
            return "Unknown source"
        
        parts = []
        
        # Handle different possible metadata keys
        if 'filename' in metadata:
            parts.append(f"File: {metadata['filename']}")
        elif 'file' in metadata:
            parts.append(f"File: {metadata['file']}")
        elif 'document' in metadata:
            parts.append(f"Document: {metadata['document']}")
        
        if 'page' in metadata:
            parts.append(f"Page: {metadata['page']}")
        elif 'page_number' in metadata:
            parts.append(f"Page: {metadata['page_number']}")
        
        if 'chunk_index' in metadata:
            parts.append(f"Section: {metadata['chunk_index']}")
        elif 'chunk' in metadata:
            parts.append(f"Section: {metadata['chunk']}")
        
        if 'source' in metadata:
            parts.append(f"Source: {metadata['source']}")
        
        # If no standard fields found, show first few available keys
        if not parts:
            available_keys = list(metadata.keys())[:3]  # Show first 3 keys
            for key in available_keys:
                if isinstance(metadata[key], (str, int, float)):
                    parts.append(f"{key}: {metadata[key]}")
        
        return " | ".join(parts) if parts else "Unknown source"
    
    #retriever
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
    
    #TODO: image output error; formatting required?
    def answer_question(self, question: str, search_results: List[Dict]) -> Dict[str, Any]:
        """
        Answer question using provided search results (for API compatibility).
        
        Args:
            question: User's question
            search_results: Pre-retrieved search results from vector store
            
        Returns:
            Answer with source citations
        """
        if not self.qa_chain:
            return {
                'answer': "QA chain not initialized.",
                'sources': [],
                'question': question
            }
        
        try:
            # Debug: Log search results structure
            logger.info(f"Processing {len(search_results)} search results")
            if search_results:
                logger.info(f"Sample result keys: {list(search_results[0].keys())}")
            
            # Convert search results to LangChain Document format
            docs = []
            for i, result in enumerate(search_results):
                try:
                    # Handle different result formats
                    content = result.get('content', result.get('text', result.get('page_content', '')))
                    if not content:
                        logger.warning(f"No content found in result {i}: {result}")
                        continue
                    
                    # Safely handle metadata
                    metadata = result.get('metadata', {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    # Add score to metadata if available
                    if 'score' in result:
                        metadata['similarity_score'] = result['score']
                    
                    # Add other fields that might be useful
                    for key in ['filename', 'file', 'page', 'source', 'chunk_index']:
                        if key in result and key not in metadata:
                            metadata[key] = result[key]
                    
                    doc = Document(page_content=content, metadata=metadata)
                    docs.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing search result {i}: {str(e)}")
                    logger.error(f"Result data: {result}")
                    continue
            
            if not docs:
                return {
                    'answer': "No relevant documents found for your question.",
                    'sources': [],
                    'question': question
                }
            
            logger.info(f"Created {len(docs)} documents for QA chain")
            
            # Use QA chain to answer based on provided documents
            result = self.qa_chain({
                "input_documents": docs,
                "question": question
            })
            
            # Format response for API compatibility
            response = {
                'answer': result.get('output_text', 'No answer generated'),
                'sources': self._format_sources_for_api(docs, search_results),
                'question': question
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'question': question
            }
    
    def _format_sources_for_api(self, docs: List[Document], search_results: List[Dict]) -> List[Dict[str, Any]]:
        """Format sources for API response."""
        sources = []
        
        for i, (doc, result) in enumerate(zip(docs, search_results)):
            try:
                source = {
                    'index': i + 1,
                    'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': result.get('score', 0.0),
                    'source': self._format_source_info(doc.metadata)
                }
                sources.append(source)
            except Exception as e:
                logger.error(f"Error formatting source {i}: {str(e)}")
                # Add a minimal source entry to avoid breaking the response
                sources.append({
                    'index': i + 1,
                    'content': str(doc.page_content)[:300] + "..." if len(str(doc.page_content)) > 300 else str(doc.page_content),
                    'metadata': {},
                    'similarity_score': 0.0,
                    'source': f"Source {i + 1}"
                })
        
        return sources
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about the QA chain."""
        stats = {
            'qa_chain_initialized': self.qa_chain is not None,
            'vector_store_loaded': self.vector_store is not None,
            'llm_model': getattr(self.llm, 'model', 'unknown'),
            'embedding_model': getattr(self.embeddings, 'model_name', 'unknown'),
            'prompt_template': str(self.prompt_template.template) if self.prompt_template else "default"
        }
        
        if self.vector_store and hasattr(self.vector_store, 'index'):
            stats['total_documents'] = self.vector_store.index.ntotal
        
        return stats