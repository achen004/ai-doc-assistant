"""Gradio user interface for the AI document assistant."""

import gradio as gr
import requests
import json
from typing import List, Dict, Any, Optional
import logging
import os

qa_chain=None
vector_store=None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint configuration
API_BASE_URL = "http://localhost:8000" #adjust accordingly

def format_response_with_images(result):
    """Format API response to include images in Gradio."""
    answer = result["answer"]
    sources = result.get("sources", [])
    
    if not sources:
        return answer
    
    # Add text sources
    text_sources = []
    image_sources = []
    
    for source in sources:
        if any(key in source for key in ['image', 'image_path', 'image_url']):
            image_sources.append(source)
        else:
            text_sources.append(source)
    
    # Format text sources
    if text_sources:
        answer += "\n\n**📚 Text Sources:**\n"
        for i, source in enumerate(text_sources[:3], 1):
            score = source.get('similarity_score', 0)
            source_info = source.get('source', 'Unknown')
            answer += f"{i}. {source_info} (Score: {score:.3f})\n"
    
    # Format image sources
    if image_sources:
        answer += "\n\n**🖼️ Related Images:**\n"
        for i, source in enumerate(image_sources[:3], 1):
            score = source.get('similarity_score', 0)
            source_info = source.get('source', 'Unknown')
            answer += f"{i}. {source_info} (Score: {score:.3f})\n"
            
            # Try to include image if path exists
            image_path = source.get('image_path') or source.get('image') or source.get('image_url')
            if image_path:
                answer += f"   📎 Image: {image_path}\n"
    
    return answer

def answer_question(question: str) -> str:
    """
    Simple function to answer questions using the QA chain.
    
    Args:
        question: User's question
        
    Returns:
        Answer from the AI assistant
    """
    if not question.strip():
        return "Please enter a question."
    
    try:
        payload = {
            "question": question,
            "search_type": "both",  # Changed from "text" to "both" for images
            "k": 5
        }
        
        response = requests.post(f"{API_BASE_URL}/question", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Debug: Log what we're getting
            print(f"API Response: {result}")  # Debug
            
            answer = result["answer"]
            
            # Check if sources contain images
            if "sources" in result and result["sources"]:
                print(f"Found {len(result['sources'])} sources")  # Debug
                for i, source in enumerate(result["sources"]):
                    print(f"Source {i}: {source.keys()}")  # Debug
                    if 'image' in source or 'image_path' in source:
                        print(f"Found image in source {i}")  # Debug
            
            return format_response_with_images(result)
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def upload_pdf(file):
    """
    Upload a PDF file to the backend.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Status message
    """
    if file is None:
        return "No file selected."
    
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f, "application/pdf")}
            response = requests.post(f"{API_BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            return f"✅ Successfully processed {file.name}\n" + \
                   f"📄 Pages: {result['pages']}\n" + \
                   f"📝 Text chunks: {result['text_chunks']}\n" + \
                   f"🖼️ Images: {result['images']}"
        else:
            return f"❌ Error: {response.json().get('detail', 'Unknown error')}"
    
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        return f"❌ Error uploading file: {str(e)}"


def ask_question(question: str, search_type: str, k: int):
    """
    Ask a question to the AI assistant.
    
    Args:
        question: User's question
        search_type: Type of search (text, image, both)
        k: Number of results to retrieve
        
    Returns:
        Answer and sources
    """
    if not question.strip():
        return "Please enter a question.", ""
    
    try:
        payload = {
            "question": question,
            "search_type": search_type,
            "k": k
        }
        
        response = requests.post(f"{API_BASE_URL}/question", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Format answer
            answer = result["answer"]
            
            # Format sources
            sources = result["sources"]
            sources_text = "\n\n**Sources:**\n"
            
            for i, source in enumerate(sources, 1):
                sources_text += f"{i}. **{source['filename']}** (Page {source['page']}) - Score: {source['score']:.3f}\n"
                sources_text += f"   Preview: {source['text_preview']}\n\n"
            
            return answer, sources_text if sources else "No sources found."
        
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            return f"❌ Error: {error_msg}", ""
    
    except Exception as e:
        logger.error(f"Error asking question: {str(e)}")
        return f"❌ Error: {str(e)}", ""

def ask_question_with_images(question: str, search_type: str, k: int):
    """
    Ask question and return answer, sources, and images separately.
    """
    if not question.strip():
        return "Please enter a question.", "", []
    
    try:
        payload = {
            "question": question,
            "search_type": search_type,
            "k": k
        }
        
        response = requests.post(f"{API_BASE_URL}/question", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract answer
            answer = result["answer"]
            
            # Extract sources and images
            sources_text = ""
            image_paths = []
            text_sources = []
            
            for source in result.get("sources", []):
                score = source.get('similarity_score', 0)
                source_info = source.get('source', 'Unknown')
                
                # Check if this source contains an image
                image_path = source.get('image_path') or source.get('image') or source.get('image_url')
                
                if image_path and search_type in ["image", "both"]:
                    # This is an image source
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                    sources_text += f"🖼️ Image: {source_info} (Score: {score:.3f})\n"
                    sources_text += f"   Path: {image_path}\n\n"
                else:
                    # This is a text source
                    text_sources.append(source)
                    content_preview = source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', '')
                    sources_text += f"📄 Text: {source_info} (Score: {score:.3f})\n"
                    sources_text += f"   Preview: {content_preview}\n\n"
            
            return answer, sources_text, image_paths[:5]  # Limit to 5 images
            
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ API Error ({response.status_code}): {error_detail}", "", []
    
    except Exception as e:
        return f"❌ Error: {str(e)}", "", []


def get_system_stats():
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            return f"📊 **System Statistics:**\n" + \
                   f"• Text embeddings: {stats['text_embeddings']}\n" + \
                   f"• Image embeddings: {stats['image_embeddings']}\n" + \
                   f"• Total embeddings: {stats['total_embeddings']}"
        else:
            return "❌ Error fetching statistics"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="AI Document Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 AI Document Assistant")
        gr.Markdown("Upload PDF documents and ask questions about their content using multimodal AI.")
        
        with gr.Tab("📄 Upload Documents"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("Upload & Process", variant="primary")
                
                with gr.Column(scale=1):
                    upload_status = gr.Textbox(
                        label="Status",
                        lines=5,
                        interactive=False
                    )
            
            upload_btn.click(
                fn=upload_pdf,
                inputs=[file_input],
                outputs=[upload_status]
            )
        
        with gr.Tab("❓ Ask Questions"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your uploaded documents...",
                        lines=3
                    )
                    
                    ask_btn = gr.Button("Ask Question", variant="primary")
                
                with gr.Column(scale=3):
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=15,
                        interactive=False
                    )
            
            ask_btn.click(
                fn=answer_question,
                inputs=[question_input],
                outputs=[answer_output]
            )
        
        with gr.Tab("🔍 Advanced Search"):
            with gr.Row():
                with gr.Column(scale=2):
                    advanced_question = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your uploaded documents...be specific",
                        lines=3
                    )
                    
                    with gr.Row():
                        search_type = gr.Radio(
                            choices=["text", "image", "both"],
                            value="both",  # Changed default to "both"
                            label="Search Type"
                        )
                        k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Results"
                        )
                    
                    advanced_ask_btn = gr.Button("Ask Question", variant="primary")
                
                with gr.Column(scale=3):
                    advanced_answer_output = gr.Textbox(
                        label="Answer",
                        lines=8,
                        interactive=False
                    )
                    
                    sources_output = gr.Textbox(
                        label="Sources",
                        lines=6,
                        interactive=False
                    )
                    
                    # Add image gallery for displaying found images
                    images_output = gr.Gallery(
                        label="Related Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        height="300px",
                        allow_preview=True
                    )
            
            # Update the click handler
            advanced_ask_btn.click(
                fn=ask_question_with_images,  # Use the new function
                inputs=[advanced_question, search_type, k_slider],
                outputs=[advanced_answer_output, sources_output, images_output]  # Add images_output
            )
        
        with gr.Tab("📊 Statistics"):
            with gr.Row():
                stats_btn = gr.Button("Refresh Statistics", variant="secondary")
                stats_output = gr.Textbox(
                    label="System Statistics",
                    lines=5,
                    interactive=False
                )
            
            stats_btn.click(
                fn=get_system_stats,
                outputs=[stats_output]
            )
        
        # Load statistics on startup
        demo.load(fn=get_system_stats, outputs=[stats_output])
        
        gr.Markdown("""
        ## 🚀 How to Use:
        1. **Upload**: Upload PDF documents using the Upload tab
        2. **Ask**: Ask specific questions about your documents in the Questions tab
        3. **Advanced Search**: Use the Advanced Search tab for more control
        4. **View Stats**: Check system statistics in the Statistics tab
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,    #change as necessary
        share=False,
        debug=True
    )