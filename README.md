# Doc-Rag-System

An intelligent **Document Retrieval-Augmented Generation (RAG)** system built with **FastAPI, FAISS, and OpenAI**.  
It allows users to upload documents (PDF, DOCX, TXT), query them, and receive accurate, contextual answers with proper citations.

---

## 🚀 Features
- Upload and embed documents (`/api/embedding`)
- Semantic search using **Sentence Transformers** (`all-mpnet-base-v2`)
- Vector storage with **FAISS**
- Reranking using **Cross-Encoders**
- Query with **citations** and conversation history (`/api/query`)
- Modular codebase for easy extension
- Supports **PDF, DOCX, and TXT** documents

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/doc-rag-system.git
cd doc-rag-system 
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
Create a .env file in the project root:
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the API Server
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## ⚙️ Technology Choice Justifications

- FastAPI → High-performance, async web framework, great for ML APIs.

- Sentence Transformers (all-mpnet-base-v2) → Provides strong semantic embeddings with 768-dim vectors.

- FAISS (Facebook AI Similarity Search) → Efficient similarity search over vector embeddings.

- Cross-Encoder Reranker → Improves retrieval precision by reranking top-k results.

- OpenAI GPT-4o-mini → Generates fluent, contextual answers grounded in retrieved documents.

- Modular architecture → Code separated into extractor, chunker, embedder, faiss_store, and reranker for maintainability.


## 📚 API Documentation

### 1️⃣ Embed Document
```bash
    Endpoint:

    POST /api/embedding


    Request: (multipart file upload)

    curl -X POST "http://127.0.0.1:8000/api/embedding" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@sample_docs/demo.pdf"


    Response:

    {
    "status": "success",
    "message": "Document embedded successfully.",
    "document_id": "d0c9d092-7877-424b-b63c-e770f50d95fa"
    }
```

### 2️⃣ Query Documents
```bash
    Endpoint:

    POST /api/query


    Request:

    {
    "query": "How much experience does Rahul Chotaliya have?",
    "document_id": "d0c9d092-7877-424b-b63c-e770f50d95fa",
    "conversation_id": "string",
    "require_citations": true
    }


    Response:

    {
    "status": "success",
    "response": {
        "answer": "Rahul Chotaliya has 6 years of experience as a Python Developer.",
        "citations": [
        {
            "document_name": "Rahul chotliya CV.pdf",
            "page": 1
        }
        ]
    },
    "conversation_id": "a23f9c3d-5f8b-4a6d-b8db-213b62f1efb9"
    }
```