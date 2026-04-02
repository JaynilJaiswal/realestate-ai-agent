# backend.Dockerfile
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code into the container
COPY backend/ ./backend/

# PRE-BAKE THE MODEL WEIGHTS INTO THE IMAGE
# This prevents runtime downloads and Hugging Face rate limit warnings
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose the port the FastAPI server runs on
EXPOSE 8000

# Run the FastAPI server using the absolute module path
CMD ["python", "-m", "backend.main"]