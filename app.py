import os
import uuid
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain_groq import ChatGroq

# Your existing classes
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from typing import List, Dict, Any

load_dotenv()

app = Flask(__name__)
DEFAULT_MODEL = "llama-3.1-8b-instant"

# ── Load vector store and embedding model at startup ──────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="../data/vector_store")
collection = chroma_client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)
print(f"[INFO] Vector store loaded. Documents in collection: {collection.count()}")

# ── Retriever ─────────────────────────────────────────────────────────────────
def retrieve_context(query: str, top_k: int = 5) -> str:
    """Retrieve relevant chunks from ChromaDB for a given query"""
    query_embedding = embedding_model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if not results["documents"] or not results["documents"][0]:
        return ""

    return "\n\n".join(results["documents"][0])

# ── LLM call ──────────────────────────────────────────────────────────────────
def ask_llm(question: str, context: str, model_name: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    prompt = (
        "You are a helpful assistant. Answer using only the provided context.\n"
        "If context is insufficient, say so clearly.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    llm = ChatGroq(groq_api_key=api_key, model_name=model_name)
    response = llm.invoke(prompt)
    return response.content

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "documents_in_store": collection.count()}

@app.post("/ask")
def ask_question():
    request_id = str(uuid.uuid4())
    data = request.get_json()

    if not data or not data.get("question"):
        return jsonify({"error": "question is required", "request_id": request_id}), 400

    question = data["question"].strip()
    top_k = data.get("top_k", 5)
    model = data.get("model", DEFAULT_MODEL)

    # Retrieve context from vector store automatically
    context = retrieve_context(question, top_k=top_k)

    if not context:
        return jsonify({
            "request_id": request_id,
            "question": question,
            "answer": "No relevant information found in the documents."
        })

    try:
        answer = ask_llm(question=question, context=context, model_name=model)
    except Exception as exc:
        return jsonify({"error": str(exc), "request_id": request_id}), 500

    return jsonify({
        "request_id": request_id,
        "model": model,
        "question": question,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)