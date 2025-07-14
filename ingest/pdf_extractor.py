"""PDF text and image extraction module."""

import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from PIL import Image
import io
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text and images from PDF documents."""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF file page by page.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of pages with text and metadata
        """
        doc = fitz.open(file_path)
        pages = []
        
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append({
                "file": file_path,
                "page": i + 1,
                "text": text
            })
        
        doc.close()
        return pages
    
    def extract_images(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF and save them to output directory.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of image metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        image_metadata = []
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)
            
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_filename = f"{output_dir}/{page_index}_{img_index}.png"
                
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_metadata.append({
                    "file": pdf_path,
                    "page": page_index + 1,
                    "image_path": image_filename,
                    "image_index": img_index,
                    "width": base_image.get("width", 0),
                    "height": base_image.get("height", 0)
                })
        
        doc.close()
        return image_metadata
    
    def extract_text_and_images(self, pdf_path: str, output_dir: str = "data/images") -> Dict[str, Any]:
        """
        Extract both text and images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            Dictionary containing extracted text and images by page
        """
        result = {
            'text_pages': self.extract_text(pdf_path),
            'image_metadata': self.extract_images(pdf_path, output_dir),
            'metadata': {}
        }
        
        try:
            doc = fitz.open(pdf_path)
            result['metadata'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'pages': doc.page_count,
                'file_path': pdf_path
            }
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF {pdf_path}: {str(e)}")
            
        return result