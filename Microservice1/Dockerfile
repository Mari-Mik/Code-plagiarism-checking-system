# Use a Python base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Ensure the FAISS index file is copied
COPY PY_index.faiss /app/PY_index.faiss

# Copy the requirements.txt file to the container
COPY requirements.txt /app/requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --timeout=300 -r /app/requirements.txt

# Install dependencies from requirements.txt
#RUN pip install -r /app/requirements.txt

# Expose the port
EXPOSE 5001

# Run the embedding server
CMD ["uvicorn", "embedding_server:app", "--host", "127.0.0.1", "--port", "5001", "--reload"]
