import faiss
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
import os

# FastAPI app
app = FastAPI()

# Define FAISS index paths
FAISS_INDEX_PATHS = {
    "python": "/app/PY_index.faiss",
    "html": "/app/HTML_index.faiss",
    "c": "/app/C_index.faiss"
}

# Function to load the FAISS index based on code type
def load_faiss_index(code_type: str):
    if code_type in FAISS_INDEX_PATHS:
        index_path = FAISS_INDEX_PATHS[code_type]
        if os.path.exists(index_path):
            print(f"✅ Loading {code_type} index from {index_path}...")
            return faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"❌ {code_type} index not found at {index_path}")
    else:
        raise ValueError(f"❌ Unknown code type: {code_type}")

# Request model for code snippet
class CodeSnippet(BaseModel):
    code: str
    code_type: str  # Add code_type to determine which index to use

# Function to get embeddings from the embedding server
def get_embedding_from_server(code):
    response = requests.post("http://embedding-server:5001/embed/", json={"code": code})
    if response.status_code == 200:
        return np.array(response.json()["embedding"]).astype("float32")
    else:
        raise Exception("Error fetching embedding from embedding server")

# Function to find similar code using the correct FAISS index
def find_similar_code(embedding, code_type: str, k: int = 3):
    # Load the corresponding FAISS index based on code type
    index = load_faiss_index(code_type)
    embedding = np.array(embedding).astype('float32').reshape(1, -1)  # Reshape to match index dimensionality
    _, indices = index.search(embedding, k)  # Search for the top k most similar code snippets
    return indices.tolist()

# Endpoint to check plagiarism
@app.post("/check_plagiarism/")
def check_plagiarism(snippet: CodeSnippet, k: int = 3):
    # Get the embedding for the provided code snippet
    embedding = get_embedding_from_server(snippet.code)
    
    # Find similar code using the corresponding FAISS index
    similar_code_indices = find_similar_code(embedding, snippet.code_type, k)
    
    # Get similar code files from the FAISS index (Here, you may have metadata associated with your index files)
    similar_code_files = [f"path/to/similar/code{index}" for index in similar_code_indices]
    
    return {"similar_code_indices": similar_code_indices, "similar_code_files": similar_code_files}

# To run the app: uvicorn plagiarism_check_api:app --reload
# uvicorn embedding_server:app --host 127.0.0.1 --port 8000 --reload
