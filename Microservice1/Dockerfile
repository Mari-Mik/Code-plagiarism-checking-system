# Use a Python base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch, Faiss, and other dependencies for the embedding server
RUN pip install torch faiss-cpu numpy transformers

# Install other Python dependencies from requirements.txt (if any)
RUN pip install -r requirements.txt

# Expose the port that the embedding server runs on
EXPOSE 5001

# Command to run the embedding server (adjust this to your actual script name)
CMD ["python", "embedding_server.py"]
