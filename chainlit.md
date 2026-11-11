# Groq Document Chat with Qdrant

## Start the application

Install dependencies and start Chainlit:

```powershell
python -m pip install -r requirements.txt
chainlit run app.py
```

## Configure Qdrant Cloud

Set these variables in the local `.env` file:

```text
GROQ_API_KEY=...
TAVILY_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

The application uses the existing `chatbot_documents` collection with
`BAAI/bge-small-en-v1.5` embeddings, 384 dimensions, and cosine similarity.
It verifies the collection at startup and does not recreate or clear it.

## Upload and ask questions

Upload PDF, DOCX, or TXT files from the Chainlit message composer. PDF text is
extracted page by page with PyMuPDF, DOCX files are read paragraph by paragraph,
and TXT files are read as UTF-8. Scanned PDFs are reported as unsupported
because this basic example does not include OCR.

Each upload is split into 1,000-character chunks with 150-character overlap.
Chunks are embedded and stored in Qdrant with filename, page, document, and
Chainlit session metadata. Follow-up questions use semantic retrieval of the
top four chunks from the current session only. Multiple documents can be
uploaded in one session.

The PowerShell terminal prints labelled logs for upload, extraction, chunking,
Qdrant indexing, semantic retrieval, Groq requests, tool calls, and Groq
responses. API keys are never printed.
