from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np


# FastAPI app
app = FastAPI()

# Load CodeBERT model
model = RobertaModel.from_pretrained("microsoft/codebert-base")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Embedding request model
class CodeSnippet(BaseModel):
    code: str

# Compute embedding function
def compute_embedding(code: str):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

# Endpoint to compute embedding
@app.post("/embed/")
def embed_code(snippet: CodeSnippet):
    embedding = compute_embedding(snippet.code)
    return {"embedding": embedding}

# To run the app: uvicorn embedding_server:app --reload
# uvicorn embedding_server:app --host 127.0.0.1 --port 5001 --reload