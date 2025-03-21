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

# Install dependencies from requirements.txt
RUN pip install -r /app/requirements.txt

# Expose the port that the plagiarism checking API runs on
EXPOSE 8000

# Command to run the plagiarism checking API
CMD ["uvicorn", "plagiarism_check_api:app", "--host", "127.0.0.1", "--port", "8000"]
