# 🔍 Code Plagiarism Checking System

## 📌 Project Overview

This is a **Code Plagiarism Checking System** designed to detect whether a given code snippet is plagiarized by comparing it with code from existing public repositories. The system uses **vector embeddings** to search for semantically similar code and **LLMs (Large Language Models)** to decide whether the submitted code is plagiarized.

## 🎯 Objective

The system demonstrates a simplified plagiarism checking workflow, including:
- Repository extraction
- Code indexing and embedding
- Vector similarity search
- LLM-based final plagiarism decision

---

## 🧱 Project Structure

code-plagiarism-checker/
│
├── Microservice1/                      ← Handles repository downloading, embedding, and FAISS storage
│   ├── Get_repos.py                    ← Clones code repositories from GitHub
│   ├── embedding_server.py             ← Generates embeddings using a model   (CodeBERT)
│   ├── Store_in_Faiss.py               ← Stores embeddings in a FAISS vector database
│   ├── Dockerfile                      ← Docker setup for Microservice 1
│   └── requirements.txt                ← Python dependencies for Microservice 1
│
├── Microservice2/                      ← Provides the plagiarism-checking API
│   ├── plagiarism-check-api.py         ← FastAPI server that checks plagiarism
│   ├── Dockerfile                      ← Docker setup for Microservice 2
│   └── requirement.txt                 ← Python dependencies for Microservice 2
│
├── docker-compose.yml                  ← Orchestrates both microservices using Docker
└── README.md                           ← Project documentation


---

## ⚙️ How It Works

### 1. 🗂️ Repository Extraction
- Clone 3 simple open-source repositories  using the GitHub API.
- Place them in the `clone repos/` folder.

### 2. 🧠 Code Indexing
- Files with extensions like `.py`, `.java`, `.c` etc. are parsed.
- Their code is embedded using a **pretrained model from Hugging Face**, such as `codebert-base`.
- Embeddings are stored in a vector database ( **FAISS**).

### 3. 🧾 Plagiarism Detection API
- Built using **FastAPI**.
- Accepts a code snippet (as string) via POST request.
- Computes the embedding for the submitted snippet.
- Searches the vector database for the most similar code files.
- Sends an aggregated prompt (user code + top similar files) to an **LLM** via API.
- LLM replies with `"yes"` or `"no"` to indicate plagiarism and may list the file references if plagiarism is detected.

---

## 🔧 Setup Instructions

### ✅ Prerequisites
- Python 3.9+
- Git
- Hugging Face `transformers`
- FastAPI
- FAISS
- OpenAI or Hugging Face API access (for LLM)

