"""Universal document extractor for PDF and Word documents."""

import os
import logging
from typing import Dict, Any, Union
from .pdf_extractor import PDFExtractor
from .word_extractor import WordExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Universal document extractor that handles multiple file formats."""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.word_extractor = WordExtractor()
        
        # Supported file formats
        self.supported_formats = ['.pdf', '.doc', '.docx']
    
    def extract_text_and_images(self, file_path: str, output_dir: str = "data/images") -> Dict[str, Any]:
        """
        Extract text and images from various document formats.
        
        Args:
            file_path: Path to the document
            output_dir: Directory to save extracted images
            
        Returns:
            Dictionary containing extracted text and images
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {self.supported_formats}")
        
        logger.info(f"Extracting from {file_ext} file: {file_path}")
        
        try:
            if file_ext == '.pdf':
                return self.pdf_extractor.extract_text_and_images(file_path, output_dir)
            elif file_ext in ['.doc', '.docx']:
                return self.word_extractor.extract_text_and_images(file_path, output_dir)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {str(e)}")
            raise
    
    def get_supported_formats(self) -> list:
        """Get list of supported file formats."""
        return self.supported_formats
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file format is supported."""
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.supported_formats