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

# Install FastAPI, Uvicorn, PyTorch, Faiss, and other dependencies
#RUN pip install fastapi uvicorn requests torch faiss-cpu numpy

# Install other Python dependencies from requirements.txt (if any)
RUN pip install -r requirements.txt

# Expose the port that the plagiarism checking API runs on
EXPOSE 8000

# Command to run the plagiarism checking API
CMD ["uvicorn", "plagiarism_check_api:app", "--reload"]
