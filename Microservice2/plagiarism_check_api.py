import faiss
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np


# FastAPI app
app = FastAPI()

# Load FAISS index
index = faiss.read_index("PY_index.faiss")
embeddings_dim = index.d  # Get the dimensionality of the embeddings

# OpenAI API (Use your API key)
openai_api_key = "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" 

# Request model
class CodeSnippet(BaseModel):
    code: str



# Function to call the embedding server
def get_embedding_from_server(code):
    response = requests.post("http://embedding-server:5001/compute_embedding/", json={"code": code})
    if response.status_code == 200:
        return np.array(response.json()["embedding"]).astype('float32')
    else:
        raise Exception("Error fetching embedding from embedding server")
    
    
# Function to find similar code snippets using FAISS
def find_similar_code(embedding):
    embedding = np.array(embedding).astype('float32').reshape(1, embeddings_dim)  # Reshape to match the index dimensionality
    _, indices = index.search(embedding, k=3)  # Search for top 3 most similar code snippets
    return indices.tolist()

# Call OpenAI API to evaluate plagiarism
def check_plagiarism_with_openai(code, similar_code_files):
    context = "\n".join([f"Code snippet:\n{code}\n\nSimilar code:\n{similar_code}" for similar_code in similar_code_files])
    prompt = f"Given the following code and similar code snippets, is the code plagiarized? Answer with 'Yes' or 'No'. Also provide the references."
    
    response = requests.post(
        "https://api.openai.com/v1/completions",
        headers={"Authorization": f"Bearer {openai_api_key}"},
        json={
            "model": "text-davinci-003",  # Or any LLM model you're using
            "prompt": prompt,
            "max_tokens": 50,
        },
    )
    return response.json()

@app.post("/check_plagiarism/")
def check_plagiarism(snippet: CodeSnippet):
    # Call embedding server to get embedding for the code
    embedding = get_embedding_from_server(snippet.code)
    similar_code_indices = find_similar_code(embedding)
    
    # Get similar code from the vector base (you can extract file paths or use metadata if necessary)
    similar_code_files = ["path/to/similar/code1", "path/to/similar/code2"]  # Use the actual paths from FAISS index

    # Check with OpenAI if the code is plagiarized
    plagiarism_result = check_plagiarism_with_openai(snippet.code, similar_code_files)
    
    # Extract "Yes" or "No" response
    is_plagiarized = "Yes" if "Yes" in plagiarism_result['choices'][0]['text'] else "No"
    return {"plagiarism": is_plagiarized, "references": similar_code_files}

# uvicorn plagiarism_check_api:app --reload
