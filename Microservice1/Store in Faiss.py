import os
import torch
import faiss
from transformers import RobertaTokenizer, RobertaModel
import numpy as np

# Define the code file extensions you're interested in
CODE_EXTENSIONS = ['.py', '.java', '.c', '.cpp', '.js', '.html', '.css', '.rb', '.go', '.sh']

# Function to load the CodeBERT model and tokenizer
def load_codebert_model():
    model_name = "microsoft/codebert-base"  # CodeBERT model from Hugging Face
    model = RobertaModel.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to compute embeddings for a given code snippet
def compute_embedding(code, model, tokenizer):
    # Tokenize the code snippet
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Ensure the model is in evaluation mode and run the forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The last hidden state from the model is the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling of token embeddings
    return embeddings.squeeze().cpu().numpy()

# Function to read code from a file
def read_code_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        code = file.read()
    return code

# Function to find code files in a directory
def find_code_files(directory):
    code_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in CODE_EXTENSIONS):
                code_files.append(os.path.join(root, file))
    return code_files

# Function to create and index the embeddings in FAISS
def create_faiss_index(embeddings, file_paths):
    # Convert embeddings list to a numpy array
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Initialize the FAISS index (use L2 distance metric)
    index = faiss.IndexFlatL2(embeddings_np.shape[1])  # Vector dimension must match embeddings size
    index.add(embeddings_np)  # Add embeddings to the index
    
    return index, embeddings_np

# Function to save the FAISS index to a file
def save_faiss_index(index, file_name):
    faiss.write_index(index, file_name)
    print(f"Index saved to {file_name}")

# Main function to process all code files and index them based on their extension
def main():
    base_directory = "cloned_repos"  # Folder where the repositories are cloned
    model, tokenizer = load_codebert_model()  # Load the CodeBERT model and tokenizer
    
    code_files = find_code_files(base_directory)  # Find all code files
    
    # Dictionary to hold embeddings and file paths for each file type
    file_type_embeddings = {ext: [] for ext in CODE_EXTENSIONS}  # A dictionary to store embeddings for each file type
    file_type_paths = {ext: [] for ext in CODE_EXTENSIONS}  # A dictionary to store file paths for each file type
    
    # For each code file, compute the embedding and store it by file type
    for file_path in code_files:
        print(f"Processing file: {file_path}")
        
        # Read the code from the file
        code = read_code_file(file_path)
        
        # Compute the embedding for the code file
        embedding = compute_embedding(code, model, tokenizer)
        
        # Determine the file extension (type)
        file_extension = os.path.splitext(file_path)[1].lower()  # Get the file extension (e.g., .py, .html)
        
        if file_extension in file_type_embeddings:
            file_type_embeddings[file_extension].append(embedding)
            file_type_paths[file_extension].append(file_path)
    
    # Create and save separate FAISS indexes for each file type
    for ext, embeddings in file_type_embeddings.items():
        if embeddings:  # If there are any embeddings for this file type
            print(f"Creating index for {ext} files...")
            
            # Create a FAISS index for the current file type
            index, embeddings_np = create_faiss_index(embeddings, file_type_paths[ext])
            
            # Save the FAISS index to a file named based on the file type
            save_faiss_index(index, f"{ext[1:].upper()}_index.faiss")  # Save as PY_index.faiss, HTML_index.faiss, etc.
            
            # Optionally, save the file paths corresponding to each embedding in a separate file for reference
            with open(f"{ext[1:].upper()}_file_paths.txt", "w", encoding="utf-8") as f:
                for file_path in file_type_paths[ext]:
                    f.write(f"{file_path}\n")

    print(f"Total code files indexed: {len(code_files)}")

if __name__ == "__main__":
    main()
