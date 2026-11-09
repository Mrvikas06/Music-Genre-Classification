# Step 03: Set the Base Image in the Dockerfile
FROM python:3.11-slim

# Step 04: Docker: Set WORKDIR and Copy Application Files
WORKDIR /app

# Install system audio dependencies (libsndfile for librosa/soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Step 05: Install Python Dependencies in a Docker Image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model artifacts
COPY . .

# Step 06: Finalize Dockerfile with EXPOSE and CMD
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
