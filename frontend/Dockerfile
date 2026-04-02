# frontend.Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend code
COPY frontend/ ./frontend/

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app and bind it to all network interfaces
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]