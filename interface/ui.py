"""Gradio user interface for the AI document assistant."""

import gradio as gr
import requests
import json
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint configuration
API_BASE_URL = "http://<WSLIP>:8000" #adjust accordingly


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
        payload = {"question": question}
        response = requests.post(f"{API_BASE_URL}/ask", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result["answer"]
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            return f"❌ Error: {error_msg}"
    
    except Exception as e:
        logger.error(f"Error asking question: {str(e)}")
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
                        placeholder="Ask anything about your uploaded documents...",
                        lines=3
                    )
                    
                    with gr.Row():
                        search_type = gr.Radio(
                            choices=["text", "image", "both"],
                            value="text",
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
                        lines=10,
                        interactive=False
                    )
                    
                    sources_output = gr.Textbox(
                        label="Sources",
                        lines=8,
                        interactive=False
                    )
            
            advanced_ask_btn.click(
                fn=ask_question,
                inputs=[advanced_question, search_type, k_slider],
                outputs=[advanced_answer_output, sources_output]
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
        2. **Ask**: Ask questions about your documents in the Questions tab
        3. **Advanced Search**: Use the Advanced Search tab for more control
        4. **View Stats**: Check system statistics in the Statistics tab
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )