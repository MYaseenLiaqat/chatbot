
import os
import json
import asyncio
import re
from datetime import datetime
from pathlib import Path

import chainlit as cl
import fitz
from chainlit.context import context
from docx import Document
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncGroq,
    AuthenticationError,
    RateLimitError,
)
from tavily import TavilyClient
from qdrant_service import qdrant_service

# Load API keys
load_dotenv("backend/.env")
load_dotenv(".env")

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

groq_client = AsyncGroq(api_key=groq_api_key) if groq_api_key else None
tavily = TavilyClient(api_key=tavily_key) if tavily_key else None

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024


# -----------------------------------------
# DEBUG LOGGER
# -----------------------------------------

def debug_log(title, data):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    else:
        print(data)

    print("=" * 60, flush=True)


# -----------------------------------------
# DOCUMENT CHAT / BASIC RAG
# -----------------------------------------

def extract_document_text(file_path: str, extension: str) -> tuple[list[dict], int]:
    """Extract text while retaining a useful source location for each section."""

    sections = []

    if extension == ".pdf":
        with fitz.open(file_path) as document:
            for page_number, page in enumerate(document, start=1):
                sections.append({
                    "text": page.get_text("text"),
                    "page": page_number,
                })
            page_count = len(document)
        method = "PyMuPDF page-by-page extraction"
        location_count = page_count
        location_name = "pages"
    elif extension == ".docx":
        document = Document(file_path)
        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        sections = [{"text": text, "page": None} for text in paragraphs]
        page_count = 0
        method = "python-docx paragraph extraction"
        location_count = len(paragraphs)
        location_name = "paragraphs"
    else:
        text = Path(file_path).read_text(encoding="utf-8")
        sections = [{"text": text, "page": None}]
        page_count = 0
        method = "Python UTF-8 text file reading"
        location_count = 1
        location_name = "text file"

    extracted_text = "\n".join(section["text"] for section in sections)
    debug_log("[TEXT EXTRACTION]", {
        "method": method,
        location_name: location_count,
        "extracted_text_length": len(extracted_text),
        "preview": extracted_text[:500],
    })

    return sections, page_count


def chunk_document(sections: list[dict], filename: str) -> list[dict]:
    """Split extracted sections into overlapping, searchable chunks."""

    chunks = []
    chunk_number = 1
    chunk_size = 1000
    overlap = 150

    for section in sections:
        text = re.sub(r"\s+", " ", section["text"]).strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            location = (
                f"p{section['page']:03d}"
                if section["page"] is not None
                else "section"
            )
            chunk_id = f"{Path(filename).stem}-{location}-c{chunk_number:03d}"
            chunk = {
                "id": chunk_id,
                "text": chunk_text,
                "filename": filename,
                "page": section["page"],
            }
            chunks.append(chunk)
            debug_log("[CHUNKING]", {
                "chunk_id": chunk_id,
                "chunk_length": len(chunk_text),
                "preview": chunk_text[:500],
            })
            chunk_number += 1

            if end == len(text):
                break
            start = end - overlap

    debug_log("[CHUNKING]", {
        "total_chunks": len(chunks),
        "chunk_ids": [chunk["id"] for chunk in chunks],
    })
    return chunks


def format_document_context(chunks: list[dict]) -> str:
    """Format retrieved chunks so Groq can distinguish sources from the question."""

    if not chunks:
        return "(No relevant document chunks were found.)"

    context_parts = []
    for chunk in chunks:
        page = f", page {chunk['page']}" if chunk["page"] else ""
        context_parts.append(
            f"[Source: {chunk['filename']}{page}; chunk {chunk['id']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def get_session_id() -> str:
    """Use Chainlit's stable thread ID as the Qdrant session partition."""

    return context.session.thread_id


async def process_uploaded_file(element, session_id: str) -> dict:
    """Validate, extract, chunk, and store one uploaded document."""

    filename = element.name or "uploaded_document"
    extension = Path(filename).suffix.lower()
    file_path = element.path

    if not file_path:
        raise ValueError("Chainlit did not provide a local path for the uploaded file.")

    file_size = os.path.getsize(file_path)
    debug_log("[DOCUMENT UPLOAD]", {
        "filename": filename,
        "extension": extension,
        "size_bytes": file_size,
    })

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            "Please upload a PDF, DOCX, or TXT file."
        )
    if file_size == 0:
        raise ValueError("The uploaded document is empty.")
    if file_size > MAX_DOCUMENT_SIZE:
        raise ValueError("The document is larger than the 10 MB limit.")

    sections, page_count = await asyncio.to_thread(
        extract_document_text,
        file_path,
        extension,
    )
    chunks = await asyncio.to_thread(chunk_document, sections, filename)
    if not chunks:
        if extension == ".pdf":
            raise ValueError(
                "No selectable text was found. This may be a scanned PDF; "
                "OCR is not supported in this basic version."
            )
        raise ValueError("No readable text was found in the document.")

    with open(file_path, "rb") as uploaded_file:
        document_id = qdrant_service.document_id(session_id, uploaded_file.read())

    if not cl.user_session.get("qdrant_ready", False):
        raise RuntimeError(
            "Document search is unavailable because Qdrant is not connected."
        )

    vector_count = await asyncio.to_thread(
        qdrant_service.upsert_document,
        chunks,
        session_id,
        document_id,
    )
    document_state = {
        "document_id": document_id,
        "filename": filename,
        "page_count": page_count,
        "chunks": chunks,
        "vector_count": vector_count,
    }
    documents = cl.user_session.get("documents", {})
    documents[document_id] = document_state
    cl.user_session.set("documents", documents)
    return document_state


# -----------------------------------------
# TOOL 1: CURRENT DATE AND TIME
# -----------------------------------------

def get_current_datetime() -> str:
    """Get the current local date, time, day, and timezone."""

    result = datetime.now().astimezone().strftime(
        "%A, %d %B %Y, %I:%M:%S %p %Z"
    )

    debug_log("DATE/TIME TOOL RESULT", {"result": result})
    return result


# -----------------------------------------
# TOOL 2: WEB SEARCH WITH TAVILY
# -----------------------------------------

def search_web(query: str) -> dict:
    """Search the web for current information about a topic."""

    if tavily is None:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    debug_log("TAVILY TOOL CALLED", {"query": query})

    results = tavily.search(
        query=query,
        search_depth="basic",
        max_results=5,
    )

    # Return useful source information to the model.
    formatted_results = []

    for item in results.get("results", []):
        formatted_results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        })

    output = {
        "query": query,
        "results": formatted_results,
    }

    debug_log("TAVILY SEARCH RESULTS", output)
    return output


# -----------------------------------------
# REGISTER BOTH TOOLS WITH GROQ
# -----------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": (
                "Get the actual current local date, time, day of the week, "
                "and timezone."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet using Tavily for current events, "
                "recent information, facts, or topics that need web research."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


# -----------------------------------------
# CHAINLIT CHAT
# -----------------------------------------

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("messages", [])
    cl.user_session.set("documents", {})
    cl.user_session.set("qdrant_ready", False)
    session_id = get_session_id()
    debug_log("[QDRANT]", {
        "event": "initializing",
        "session_id": session_id,
    })
    try:
        await asyncio.to_thread(qdrant_service.initialize)
        cl.user_session.set("qdrant_ready", True)
        collection_details = await asyncio.to_thread(
            qdrant_service.verify_connection
        )
        debug_log("[QDRANT]", {
            "event": "ready",
            "collection": collection_details,
        })
    except Exception as error:
        debug_log("[QDRANT]", {
            "event": "unavailable",
            "error": str(error),
        })

    await cl.Message(
        content=(
            "Hi! I'm your AI assistant. I can answer questions, "
            "check the current time, search the web, and answer questions "
            "about uploaded PDF, DOCX, or TXT documents."
        ),
        author="Groq",
    ).send()


def _assistant_message_dict(message) -> dict:
    """Convert a Groq assistant message into the history format it expects."""

    return message.model_dump(exclude_none=True)


def _tool_error_message(error: Exception) -> str:
    """Return a useful, non-sensitive tool error for the model and user."""

    return f"Tool execution failed: {error}"


async def call_groq(query: str, document_context: str | None = None):
    if groq_client is None:
        await cl.Message(
            content="Groq is not configured. Set GROQ_API_KEY in .env and restart the app.",
            author="System",
        ).send()
        return

    messages = cl.user_session.get("messages", [])
    request_text = query
    if document_context is not None:
        request_text = (
            "DOCUMENT CONTEXT (use only this retrieved context; it may be incomplete):\n"
            f"{document_context}\n\n"
            "USER QUESTION:\n"
            f"{query}\n\n"
            "Answer from the document context. If the context does not contain "
            "the answer, explicitly say that the document does not provide it. "
            "Cite the source filename and page number when available."
        )

    messages.append({"role": "user", "content": request_text})

    debug_log("[USER QUESTION]", query)
    debug_log("[LLM REQUEST]", {
        "model": MODEL,
        "user_question": query,
        "complete_document_context": document_context,
        "conversation_history": messages[:-1],
    })

    try:
        # Allow multiple tool calls before the final answer
        for step in range(5):

            debug_log("[GROQ REQUEST]", {
                "model": MODEL,
                "step": step + 1,
                "conversation_messages": len(messages),
            })

            response = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful AI assistant. Use "
                                "get_current_datetime for current date and "
                                "time questions. Use search_web for current "
                                "events, recent information, or web research. "
                                "Use tool results accurately and never invent "
                                "search results or sources."
                            ),
                        },
                        *messages,
                    ],
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=1024,
                ),
                timeout=60,
            )

            assistant_message = response.choices[0].message
            debug_log("[GROQ RESPONSE]", {
                "raw_response": response.model_dump(exclude_none=True),
                "token_usage": getattr(response, "usage", None),
            })
            messages.append(_assistant_message_dict(assistant_message))
            function_calls = assistant_message.tool_calls or []

            # No tool requested: show final answer
            if not function_calls:
                answer = assistant_message.content or "No text response received."

                debug_log("[LLM RESPONSE]", {
                    "final_generated_answer": answer,
                    "token_usage": getattr(response, "usage", None),
                })

                await cl.Message(
                    content=answer,
                    author="Groq",
                ).send()

                cl.user_session.set("messages", messages)
                return

            for call in function_calls:
                function_name = call.function.name
                raw_arguments = call.function.arguments or "{}"
                debug_log("[GROQ TOOL CALL]", {
                    "tool_call_id": call.id,
                    "function": function_name,
                    "arguments": raw_arguments,
                })

                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")

                    if function_name == "get_current_datetime":
                        result = await asyncio.to_thread(get_current_datetime)
                    elif function_name == "search_web":
                        search_query = arguments.get("query")
                        if (
                            not isinstance(search_query, str)
                            or not search_query.strip()
                        ):
                            raise ValueError("Missing or invalid search query.")
                        result = await asyncio.to_thread(search_web, search_query)
                    else:
                        raise ValueError(f"Unknown tool requested: {function_name}")
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    result = _tool_error_message(error)
                except Exception as error:
                    result = _tool_error_message(error)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
                debug_log("[TOOL RESULT]", {
                    "tool_call_id": call.id,
                    "function": function_name,
                    "result": result,
                })

            cl.user_session.set("messages", messages)

        raise RuntimeError(
            "Maximum tool steps reached without a final answer."
        )

    except AuthenticationError:
        error_message = "Groq authentication failed. Check GROQ_API_KEY."
    except RateLimitError:
        error_message = "Groq rate limit reached. Please try again later."
    except APIConnectionError:
        error_message = "Could not connect to Groq. Check your network and try again."
    except APIStatusError as error:
        error_message = f"Groq API error (status {error.status_code})."
    except APIError:
        error_message = "Groq returned an API error. Please try again."
    except Exception as error:
        error_message = f"Chatbot error: {error}"

    debug_log("[CHATBOT ERROR]", error_message)
    await cl.Message(content=error_message, author="System").send()


@cl.on_message
async def chat(message: cl.Message):
    session_id = get_session_id()
    uploaded_elements = [
        element for element in message.elements
        if getattr(element, "name", None) or getattr(element, "path", None)
    ]

    if uploaded_elements:
        if len(uploaded_elements) > 1:
            await cl.Message(
                content="Please upload one document at a time.",
                author="System",
            ).send()
            return

        try:
            document = await process_uploaded_file(
                uploaded_elements[0],
                session_id,
            )
            page_summary = (
                str(document["page_count"])
                if document["page_count"]
                else "Not applicable for DOCX/TXT"
            )
            await cl.Message(
                content=(
                    "Document uploaded successfully!\n"
                    f"Filename: {document['filename']}\n"
                    f"Pages extracted: {page_summary}\n"
                    f"Text chunks created: {len(document['chunks'])}\n"
                    f"Vectors indexed: {document['vector_count']}\n"
                    "You can now ask questions about this document."
                ),
                author="Groq",
            ).send()
        except Exception as error:
            debug_log("[DOCUMENT ERROR]", str(error))
            await cl.Message(
                content=f"Document upload error: {error}",
                author="System",
            ).send()
            return

        if not message.content.strip():
            return

    documents = cl.user_session.get("documents", {})
    document_context = None
    if documents:
        if not cl.user_session.get("qdrant_ready", False):
            await cl.Message(
                content="Document search is unavailable because Qdrant is not connected.",
                author="System",
            ).send()
            return
        try:
            started_at = asyncio.get_running_loop().time()
            retrieved_chunks = await asyncio.to_thread(
                qdrant_service.retrieve,
                message.content,
                session_id,
                4,
            )
            retrieval_duration = asyncio.get_running_loop().time() - started_at
            debug_log("[RETRIEVAL]", {
                "method": "Qdrant semantic search",
                "session_id": session_id,
                "duration_seconds": round(retrieval_duration, 4),
                "number_retrieved": len(retrieved_chunks),
                "chunks": retrieved_chunks,
            })
            document_context = format_document_context(retrieved_chunks)
        except Exception as error:
            debug_log("[QDRANT RETRIEVAL ERROR]", str(error))
            await cl.Message(
                content="Document search failed. Please try again or upload the document again.",
                author="System",
            ).send()
            return

    await call_groq(message.content, document_context)


@cl.on_chat_end
async def cleanup_chat():
    """Best-effort cleanup; session filters prevent cross-session deletion."""

    try:
        await asyncio.to_thread(qdrant_service.delete_session, get_session_id())
        debug_log("[QDRANT]", {"event": "session vectors deleted"})
    except Exception as error:
        debug_log("[QDRANT CLEANUP]", {
            "event": "cleanup skipped",
            "error": str(error),
        })
