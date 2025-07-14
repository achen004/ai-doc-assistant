# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs config

# Expose ports
EXPOSE 8000 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV OLLAMA_HOST=0.0.0.0:11434

# Create startup script
RUN echo '#!/bin/bash\\n\\
set -e\\n\\
\\n\\
# Start Ollama in the background\\n\\
ollama serve &\\n\\
\\n\\
# Wait for Ollama to be ready\\n\\
echo "Waiting for Ollama to start..."\\n\\
while ! curl -s http://localhost:11434/api/tags > /dev/null; do\\n\\
    sleep 1\\n\\
done\\n\\
\\n\\
# Pull the mistral model\\n\\
echo "Pulling Mistral model..."\\n\\
ollama pull mistral\\n\\
\\n\\
# Start the FastAPI server\\n\\
echo "Starting FastAPI server..."\\n\\
uvicorn backend.server:app --host 0.0.0.0 --port 8000 &\\n\\
\\n\\
# Start the Gradio interface\\n\\
echo "Starting Gradio interface..."\\n\\
python interface/ui.py\\n\\
' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["./start.sh"]