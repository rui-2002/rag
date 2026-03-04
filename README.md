rag_app/
├── app.py                    # Flask app factory + all API routes
├── requirements.txt
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── config.py             # All config / tuneable parameters
│   ├── logger.py             # Centralised logging (console + file)
│   ├── ingestion.py          # Load PDFs & TXTs, split into chunks
│   ├── embeddings.py         # SentenceTransformer wrapper (singleton)
│   ├── vector_store.py       # ChromaDB wrapper (CRUD, deduplication)
│   └── rag_pipeline.py       # Retrieval + LLM generation
│
├── scripts/
│   └── ingest_cli.py         # CLI tool for bulk ingestion
│
├── data/
│   ├── uploads/              # Uploaded files saved here
│   └── vector_store/         # ChromaDB persisted here
│
└── logs/
    └── rag_app.log
Run API:
```powershell
python app.py
```

Run Streamlit UI (new):
```powershell
streamlit run streamlit_app.py
```
