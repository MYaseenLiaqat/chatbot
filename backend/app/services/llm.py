from groq import AsyncGroq

from backend.app.core.config import settings


async def complete(messages: list[dict], *, json_mode: bool = False) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("Groq is not configured. Set GROQ_API_KEY.")
    client = AsyncGroq(api_key=settings.groq_api_key)
    kwargs = {
        "model": settings.groq_model,
        "messages": messages,
        "max_tokens": 1400,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def context_text(chunks: list[dict]) -> str:
    if not chunks:
        return "(No matching profile evidence was found.)"
    parts = []
    for item in chunks:
        page = f", page {item['page']}" if item.get("page") else ""
        parts.append(
            f"[{item.get('source_type', 'document')}: "
            f"{item.get('filename', '')}{page}]\n{item['text']}"
        )
    return "\n\n".join(parts)
