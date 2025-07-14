"""Vector store management using FAISS."""

import faiss
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
            # Save FAISS indices
            faiss.write_index(self.text_index, f"{self.index_path}_text.index")
            faiss.write_index(self.image_index, f"{self.index_path}_image.index")
            
            # Save metadata
            with open(f"{self.index_path}_text_metadata.pkl", "wb") as f:
                pickle.dump(self.text_metadata, f)
            
            with open(f"{self.index_path}_image_metadata.pkl", "wb") as f:
                pickle.dump(self.image_metadata, f)
            
            logger.info(f"Index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self):
        """Load the index and metadata from disk."""
        try:
            text_index_path = f"{self.index_path}_text.index"
            image_index_path = f"{self.index_path}_image.index"
            
            if os.path.exists(text_index_path):
                self.text_index = faiss.read_index(text_index_path)
                
                with open(f"{self.index_path}_text_metadata.pkl", "rb") as f:
                    self.text_metadata = pickle.load(f)
            
            if os.path.exists(image_index_path):
                self.image_index = faiss.read_index(image_index_path)
                
                with open(f"{self.index_path}_image_metadata.pkl", "rb") as f:
                    self.image_metadata = pickle.load(f)
            
            logger.info(f"Index loaded from {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # Initialize empty indices if loading fails
            self._initialize_indices()
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the index."""
        return {
            'text_embeddings': self.text_index.ntotal if self.text_index else 0,
            'image_embeddings': self.image_index.ntotal if self.image_index else 0,
            'total_embeddings': (self.text_index.ntotal if self.text_index else 0) + 
                               (self.image_index.ntotal if self.image_index else 0)
        }