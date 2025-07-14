"""Image processing and embedding module."""

from typing import List, Dict, Any
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images and generate embeddings using CLIP."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def embed_image(self, image_path: str) -> List[float]:
        """
        Generate CLIP embedding for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image embedding vector
        """
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {str(e)}")
            raise
    
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """
        Generate CLIP embeddings for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of image embedding vectors
        """
        embeddings = []
        for image_path in image_paths:
            try:
                embedding = self.embed_image(image_path)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Skipping image {image_path}: {str(e)}")
                continue
        
        return embeddings
    
    def process_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process images and generate embeddings.
        
        Args:
            images: List of image dictionaries with PIL images
            
        Returns:
            List of processed images with embeddings
        """
        processed_images = []
        
        for img_data in images:
            try:
                # Preprocess image
                image = img_data['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Generate embedding
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    embedding = image_features.cpu().numpy().flatten().tolist()
                
                processed_images.append({
                    'embedding': embedding,
                    'source': img_data,
                    'width': img_data['width'],
                    'height': img_data['height'],
                    'page': img_data['page'],
                    'index': img_data['index']
                })
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                continue
        
        return processed_images
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate CLIP text embedding for image search.
        
        Args:
            text: Query text
            
        Returns:
            Text embedding vector
        """
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                embedding = text_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise