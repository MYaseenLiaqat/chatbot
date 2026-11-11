# ProfilePilot AI

ProfilePilot AI is a small, temporary-session career assistant. A visitor can
upload a PDF CV, import up to ten public GitHub repositories, ask grounded
questions, and compare their profile with a pasted job description.

## Features

- PDF-only CV validation, PyMuPDF extraction, chunking, FastEmbed embeddings
- Public GitHub REST API import with README content and no OAuth
- Session-scoped semantic retrieval and source references
- Groq-powered chat and structured job analysis
- Reset endpoint that deletes only the current session's Qdrant vectors
- Existing Chainlit document chat remains available through `app.py`

## Architecture

`frontend/` is a small Next.js client. `backend/app/main.py` exposes FastAPI
endpoints and keeps temporary session metadata in memory. Qdrant stores only
embedding payloads and is always queried with a `session_id` filter. The
existing `chatbot_documents` collection, 384-dimensional vectors, cosine
distance, and `BAAI/bge-small-en-v1.5` model are preserved.

Temporary sessions expire after 24 hours or when explicitly reset. Restarting
the backend expires in-memory session tokens; Qdrant vectors are still
protected by session filters and can be removed using the reset endpoint while
the session is active.

## Setup on Windows

From the repository root:

```powershell
Copy-Item .env.example .env
python -m pip install -r backend\requirements.txt
```

Set `GROQ_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` in `.env`. Use the
existing `chatbot_documents` collection; do not run a collection reset or
recreation. `SESSION_SECRET` is optional locally but should be a stable random
value in deployment. `FRONTEND_URL` should match the frontend origin.

Start the API:

```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

In another PowerShell terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

The original Chainlit app can still be run with:

```powershell
python -m pip install -r requirements.txt
chainlit run app.py
```

## API

- `GET /health`
- `POST /api/v1/sessions`
- `DELETE /api/v1/sessions`
- `POST /api/v1/documents/upload` with a PDF multipart file
- `POST /api/v1/github/import` with `{"username":"..."}`
- `GET /api/v1/profile`
- `POST /api/v1/chat` with `{"message":"..."}`
- `POST /api/v1/jobs/analyze` with `{"description":"..."}`

All endpoints except health and session creation use the current
`Authorization: Bearer <session_token>` header. API keys stay server-side.

## Deployment

Deploy `backend/` (or the repository root with the documented uvicorn command)
to Render or Railway using the same environment variables and a persistent
`SESSION_SECRET`. Deploy `frontend/` to Vercel with
`NEXT_PUBLIC_API_URL=https://your-backend.example.com`. Configure the backend
`FRONTEND_URL` to the Vercel origin.

## Known limitations

This is a portfolio proof of concept: sessions are intentionally temporary and
in-memory, there is no account system, scanned PDFs need OCR, GitHub data is
limited to public non-fork repositories, and a backend restart expires active
browser sessions.
