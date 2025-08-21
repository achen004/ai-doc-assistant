"""Vector store management using FAISS."""

import faiss  # Raw FAISS for custom operations
from langchain_community.vectorstores import FAISS as LangChainFAISS  # LangChain wrapper
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manage vector embeddings and similarity search using FAISS."""
    
    def __init__(self, dimension: int = 384, index_path: str = "data/vector_index"):
        self.dimension = dimension
        self.index_path = index_path
        self.text_index = None
        self.image_index = None
        self.text_metadata = []
        self.image_metadata = []
        self.langchain_vector_store = None  # For QA chain compatibility
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Initialize FAISS indices
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize FAISS indices for text and images."""
        self.text_index = faiss.IndexFlatL2(self.dimension)  # L2 distance for similarity
        self.image_index = faiss.IndexFlatL2(512)  # CLIP embeddings are 512-dimensional
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
    
    def add_text_embeddings(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """
        Add text embeddings to the index.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata for each embedding
        """
        if not embeddings:
            return
            
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.text_index.add(embeddings_array)
        self.text_metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} text embeddings to index")
    
    def add_image_embeddings(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """
        Add image embeddings to the index.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata for each embedding
        """
        if not embeddings:
            return
            
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.image_index.add(embeddings_array)
        self.image_metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} image embeddings to index")
    
    def create_langchain_vector_store(self, embedding):
        #TODO:change embedding_function to regular embedding to resolve 'HuggingFaceEmbeddings' object is not callable error
        """
        Create LangChain FAISS vector store from text index for QA chain compatibility.
        
        Args:
            embedding: LangChain embedding
        """
        if self.text_index.ntotal == 0:
            logger.warning("No text embeddings to create LangChain vector store")
            return None
        
        try:
            # Convert metadata to LangChain Documents
            documents = []
            for i, meta in enumerate(self.text_metadata):
                # Ensure we have text content
                content = meta.get('text', meta.get('content', f"Document {i}"))
                doc = Document(
                    page_content=content,
                    metadata={k: v for k, v in meta.items() if k != 'text'}
                )
                documents.append(doc)
            
            # Create LangChain FAISS from documents
            self.langchain_vector_store = LangChainFAISS.from_documents(
                documents, 
                embedding
            )
            
            logger.info("Created LangChain FAISS vector store")
            return self.langchain_vector_store
            
        except Exception as e:
            logger.error(f"Error creating LangChain vector store: {str(e)}")
            return None
    
    def search_text(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar text chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of search results with metadata and scores
        """
        if self.text_index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.text_index.search(query_array, min(k, self.text_index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                result = self.text_metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        return results
    
    def search_images(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar images.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of search results with metadata and scores
        """
        if self.image_index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.image_index.search(query_array, min(k, self.image_index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                result = self.image_metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        return results
    
    def save_index(self):
        """Save the index and metadata to disk."""
        try:
            # Save raw FAISS indices
            text_fpath = os.path.join(self.index_path, "text.faiss")
            image_fpath = os.path.join(self.index_path, "image.faiss")
            faiss.write_index(self.text_index, text_fpath)
            faiss.write_index(self.image_index, image_fpath)
            
            # Save metadata
            text_metapath = os.path.join(self.index_path, "text_metadata.pkl")
            image_metapath = os.path.join(self.index_path, "image_metadata.pkl")
            with open(text_metapath, "wb") as f:
                pickle.dump(self.text_metadata, f)
            
            with open(image_metapath, "wb") as f:
                pickle.dump(self.image_metadata, f)
            
            # Save LangChain vector store if it exists
            if self.langchain_vector_store:
                langchain_path = os.path.join(self.index_path, "langchain_store")
                self.langchain_vector_store.save_local(langchain_path)
                logger.info(f"LangChain vector store saved to {langchain_path}")
            
            logger.info(f"Index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self):
        """Load the index and metadata from disk."""
        try:
            text_index_path = os.path.join(self.index_path, "text.faiss")
            image_index_path = os.path.join(self.index_path, "image.faiss")
            text_metapath = os.path.join(self.index_path, "text_metadata.pkl")
            image_metapath = os.path.join(self.index_path, "image_metadata.pkl")
            
            if os.path.exists(text_index_path):
                self.text_index = faiss.read_index(text_index_path)
                
                with open(text_metapath, "rb") as f:
                    self.text_metadata = pickle.load(f)
            
            if os.path.exists(image_index_path):
                self.image_index = faiss.read_index(image_index_path)
                
                with open(image_metapath, "rb") as f:
                    self.image_metadata = pickle.load(f)
            
            logger.info(f"Index loaded from {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # Initialize empty indices if loading fails
            self._initialize_indices()
    
    def load_langchain_vector_store(self, embedding):
        """
        Load LangChain vector store from disk.
        
        Args:
            embedding: LangChain embedding 
            
        Returns:
            LangChain FAISS vector store or None
        """
        try:
            langchain_path = os.path.join(self.index_path, "langchain_store")
            if os.path.exists(langchain_path):
                self.langchain_vector_store = LangChainFAISS.load_local(
                    langchain_path, 
                    embedding,
                    #allow_dangerous_deserialization=True
                )
                logger.info("Loaded LangChain vector store from disk")
                return self.langchain_vector_store
            else:
                logger.info("No saved LangChain vector store found, creating new one")
                return self.create_langchain_vector_store(embedding)
                
        except Exception as e:
            logger.error(f"Error loading LangChain vector store: {str(e)}")
            return None
    
    def get_langchain_vector_store(self, embedding):
        """
        Get or create LangChain vector store for QA chain.
        
        Args:
            embedding: LangChain embedding 
            
        Returns:
            LangChain FAISS vector store
        """
        if self.langchain_vector_store is None:
            return self.load_langchain_vector_store(embedding)
        return self.langchain_vector_store
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the index."""
        return {
            'text_embeddings': self.text_index.ntotal if self.text_index else 0,
            'image_embeddings': self.image_index.ntotal if self.image_index else 0,
            'total_embeddings': (self.text_index.ntotal if self.text_index else 0) + 
                               (self.image_index.ntotal if self.image_index else 0),
            'has_langchain_store': self.langchain_vector_store is not None
        }