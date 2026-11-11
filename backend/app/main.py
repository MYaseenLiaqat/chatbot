import asyncio
import json
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import APIError
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.sessions import Session, session_store
from backend.app.services.documents import document_id, extract_pdf_chunks, index_chunks
from backend.app.services.github import import_repositories, repository_chunks
from backend.app.services.llm import complete, context_text
from qdrant_service import qdrant_service

app = FastAPI(title="ProfilePilot AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class GitHubRequest(BaseModel):
    username: str = Field(min_length=1, max_length=39)


class JobRequest(BaseModel):
    description: str = Field(min_length=20, max_length=20000)


def require_session(
    authorization: Annotated[str | None, Header()] = None,
) -> Session:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="A valid session is required.")
    session = session_store.get_from_token(authorization[7:])
    if session is None:
        raise HTTPException(status_code=401, detail="Session expired or invalid.")
    return session


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "profilepilot-api"}


@app.post("/api/v1/sessions")
async def create_session() -> dict:
    session = session_store.create()
    return {"session_token": session_store.token_for(session), "expires_in": session_store.ttl_seconds}


@app.delete("/api/v1/sessions")
async def delete_session(session: Session = Depends(require_session)) -> dict:
    try:
        if qdrant_service.client is not None:
            await asyncio.to_thread(qdrant_service.delete_session, session.session_id)
    finally:
        session_store.remove(session.session_id)
    return {"deleted": True}


@app.post("/api/v1/documents/upload")
async def upload_cv(
    file: UploadFile = File(...),
    session: Session = Depends(require_session),
) -> dict:
    if file.content_type != "application/pdf" or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF CV files are accepted.")
    data = await file.read()
    if len(data) > settings.max_upload_size:
        raise HTTPException(status_code=413, detail="The PDF must be 10 MB or smaller.")
    try:
        chunks = extract_pdf_chunks(data, file.filename)
        if qdrant_service.client is None:
            await asyncio.to_thread(qdrant_service.initialize)
        doc_id = document_id(session.session_id, data)
        count = await asyncio.to_thread(index_chunks, chunks, session.session_id, doc_id, "cv")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="CV indexing failed. Try again.") from error
    record = {"document_id": doc_id, "filename": file.filename, "chunks": count}
    session.documents = [record]
    return {"indexed": True, **record}


@app.post("/api/v1/github/import")
async def github_import(
    request: GitHubRequest,
    session: Session = Depends(require_session),
) -> dict:
    try:
        repositories = await asyncio.to_thread(import_repositories, request.username)
        if qdrant_service.client is None:
            await asyncio.to_thread(qdrant_service.initialize)
        for repository in repositories:
            chunks = repository_chunks(repository)
            repo_id = qdrant_service.document_id(
                session.session_id, repository["url"].encode()
            )
            await asyncio.to_thread(
                index_chunks, chunks, session.session_id, repo_id, "github"
            )
        session.repositories = repositories
        return {"imported": len(repositories), "repositories": repositories}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="GitHub import failed. Try again.") from error


@app.get("/api/v1/profile")
async def profile(session: Session = Depends(require_session)) -> dict:
    return {
        "ready": bool(session.documents or session.repositories),
        "documents": session.documents,
        "repositories": session.repositories,
    }


@app.post("/api/v1/chat")
async def chat(request: ChatRequest, session: Session = Depends(require_session)) -> dict:
    if qdrant_service.client is None:
        await asyncio.to_thread(qdrant_service.initialize)
    chunks = await asyncio.to_thread(
        qdrant_service.retrieve, request.message, session.session_id, 4
    )
    evidence = [chunk for chunk in chunks if chunk["score"] >= 0.25]
    history = session.history[-10:]
    messages = [{
        "role": "system",
        "content": (
            "You are ProfilePilot AI. Never invent personal facts. Use profile "
            "evidence when present and cite its source. If personal evidence is "
            "missing, say: I couldn't find this information in your uploaded "
            "profile. You may then provide a separately labeled general knowledge "
            "answer. Keep answers concise."
        ),
    }, *history, {
        "role": "user",
        "content": f"PROFILE EVIDENCE:\n{context_text(evidence)}\n\nQUESTION:\n{request.message}",
    }]
    try:
        answer = await complete(messages)
    except Exception as error:
        raise HTTPException(status_code=503, detail="The AI service is unavailable.") from error
    session.history.extend([
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": answer},
    ])
    sources = [
        {"filename": chunk["filename"], "page": chunk["page"], "type": chunk["source_type"]}
        for chunk in evidence
    ]
    return {"answer": answer, "sources": sources, "based_on_profile": bool(evidence)}


@app.post("/api/v1/jobs/analyze")
async def analyze_job(
    request: JobRequest,
    session: Session = Depends(require_session),
) -> dict:
    if qdrant_service.client is None:
        await asyncio.to_thread(qdrant_service.initialize)
    chunks = await asyncio.to_thread(
        qdrant_service.retrieve, request.description, session.session_id, 8
    )
    prompt = (
        "Compare this job description with the profile evidence. Never invent "
        "experience. Return JSON with keys: suitability, matching_skills, "
        "partial_matches, missing_or_unverified_skills, relevant_projects, "
        "experience_gaps, interview_preparation. Values should be concise arrays "
        "or a short string. Job description:\n"
        f"{request.description}\n\nProfile evidence:\n{context_text(chunks)}"
    )
    try:
        result = json.loads(
            await complete([{"role": "user", "content": prompt}], json_mode=True)
        )
    except (json.JSONDecodeError, RuntimeError, APIError) as error:
        raise HTTPException(status_code=503, detail="Job analysis failed. Try again.") from error
    return result
