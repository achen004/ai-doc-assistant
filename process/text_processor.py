"""Text processing and chunking module."""

from typing import List, Dict, Any
import re
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and chunk text for embedding generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = 300 #200-400 tokens
        self.chunk_overlap = 25 #20-50 tokens
    
    def chunk_text(self, text: str, size: int = 300, overlap: int = 25) -> List[str]:
        """
        Chunk text into smaller pieces with overlap.
        
        Args:
            text: Text to chunk
            size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunk = text[i:i + size]
            chunks.append(chunk)
        return chunks
    
    def chunk_text_with_metadata(self, text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with overlap and metadata.
        
        Args:
            text: Text to chunk
            source_info: Information about the source (file, page, etc.)
            
        Returns:
            List of text chunks with metadata
        """
        # Clean text
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'source': source_info,
                'chunk_index': 0,
                'char_start': 0,
                'char_end': len(text)
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Find the best break point
            if end < len(text):
                # Look for sentence boundaries
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundaries
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + self.chunk_size // 2:
                        end = word_end
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'source': source_info,
                    'chunk_index': chunk_index,
                    'char_start': start,
                    'char_end': end
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
        return chunks
    
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        return self.model.encode(chunks).tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        return text.strip()