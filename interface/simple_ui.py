"""
Simple Streamlit-based UI for the AI Document Assistant.
Alternative to Gradio for Windows compatibility.
"""

import streamlit as st
import requests
import json
import os
from typing import Dict, Any

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"

def upload_pdf(uploaded_file):
    """Upload PDF file to the backend."""
    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Send to backend
            with open(temp_path, "rb") as f:
                files = {"file": (uploaded_file.name, f, "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
            
            # Clean up temp file
            os.remove(temp_path)
            
            if response.status_code == 200:
                result = response.json()
                return f"✅ Successfully processed {uploaded_file.name}\n" + \
                       f"📄 Pages: {result['pages']}\n" + \
                       f"📝 Text chunks: {result['text_chunks']}\n" + \
                       f"🖼️ Images: {result['images']}"
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                return f"❌ Error: {error_msg}"
                
        except Exception as e:
            return f"❌ Error uploading file: {str(e)}"
    
    return "No file selected."

def ask_question(question: str) -> str:
    """Ask a question to the AI assistant."""
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
        return f"❌ Error: {str(e)}"

def get_stats() -> str:
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

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Document Assistant")
    st.markdown("Upload PDF documents and ask questions about their content using AI.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload Documents", "Ask Questions", "Statistics"]
    )
    
    if page == "Upload Documents":
        st.header("📄 Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to process"
        )
        
        if st.button("Upload & Process", type="primary"):
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    result = upload_pdf(uploaded_file)
                st.success(result)
            else:
                st.error("Please select a PDF file first.")
    
    elif page == "Ask Questions":
        st.header("❓ Ask Questions")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            question = st.text_area(
                "Your Question:",
                placeholder="Ask anything about your uploaded documents...",
                height=100
            )
            
            if st.button("Ask Question", type="primary"):
                if question:
                    with st.spinner("Thinking..."):
                        answer = ask_question(question)
                    
                    with col2:
                        st.subheader("Answer:")
                        st.write(answer)
                else:
                    st.error("Please enter a question.")
        
        with col2:
            if not question:
                st.info("👈 Enter your question and click 'Ask Question' to get started!")
    
    elif page == "Statistics":
        st.header("📊 System Statistics")
        
        if st.button("Refresh Statistics"):
            stats = get_stats()
            st.markdown(stats)
        
        st.markdown("Click 'Refresh Statistics' to see current system status.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ## 🚀 How to Use:
    1. **Upload**: Go to 'Upload Documents' to process PDF files
    2. **Ask**: Use 'Ask Questions' to query your documents
    3. **Stats**: Check 'Statistics' for system information
    """)

if __name__ == "__main__":
    main()