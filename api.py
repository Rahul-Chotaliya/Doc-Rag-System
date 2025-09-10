from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from embedder import Embedder
from faiss_store import FaissStore
from reranker import Reranker
from extractor import extract_pages_from_file
from chunker import chunk_text_with_overlap
from openai import OpenAI
import uuid, os, shutil
# from dotenv import 
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI(title="Doc RAG System")
embedder = Embedder("all-mpnet-base-v2")
store = FaissStore(dim=768, path="data/index.faiss", meta_path="data/meta.json")
store.load()
reranker = Reranker()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



conversations = {}

class QueryRequest(BaseModel):
    query: str
    document_id: str
    conversation_id: str = None
    require_citations: bool = True

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def generate_answer(query, docs):
    if not docs:
        return "No relevant results found in the document."
    context = "\n\n".join([d["text"] for d in docs])
    prompt = f"""
    You are an assistant. Use the following context from documents to answer the query.
    Always give factual answers and stay grounded in the context.

    Context:
    {context}

    Query: {query}

    Answer:
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating answer: {e}]"

@app.post("/api/query")
async def query_docs(req: QueryRequest):
    if not req.conversation_id or req.conversation_id == "string":
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = []
    else:
        conversation_id = req.conversation_id

    query_vector = embedder.embed_texts([req.query])[0]
    results = store.search(query_vector, top_k=3, document_id=req.document_id)
    reranked = reranker.rerank(req.query, results)

    answer = generate_answer(req.query, reranked)

    citations = [
        {"document_name": r["document_name"], "page": r["page"]}
        for r in reranked
    ] if req.require_citations else []

    conversations[conversation_id].append({"query": req.query, "answer": answer})

    return {
        "status": "success",
        "response": {
            "answer": answer,
            "citations": citations
        },
        "conversation_id": conversation_id
    }

@app.post("/api/embedding")
async def embed_document(file: UploadFile = File(...)):
    filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pages = extract_pages_from_file(filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not pages:
        raise HTTPException(status_code=400, detail="Document content is empty.")

    all_texts, all_metas = [], []
    doc_id = str(uuid.uuid4())
    for p in pages:
        chunks = chunk_text_with_overlap(p["text"], target_words=450, overlap_sentences=2)
        for idx, ch in enumerate(chunks):
            meta = {
                "document_id": doc_id,
                "document_name": os.path.basename(file.filename),
                "page": p.get("page", 1),
                "chunk_id": f"{doc_id}_{p.get('page',1)}_{idx}",
                "text": ch[:2000]
            }
            all_texts.append(ch)
            all_metas.append(meta)

    if not all_texts:
        raise HTTPException(status_code=400, detail="Document content is empty after chunking.")

    vectors = embedder.embed_texts(all_texts)
    store.add(vectors, all_metas)
    store.save()

    return {"status": "success", "message": "Document embedded successfully.", "document_id": doc_id}
