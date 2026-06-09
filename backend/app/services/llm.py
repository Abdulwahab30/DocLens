import httpx
from fastapi import HTTPException, status

from app.core.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = (
    "You are the assistant inside a Multimodal Document Intelligence app that lets users "
    "chat with their uploaded documents (PDFs, scans, invoices, reports) and cite the exact "
    "source page. If the user hasn't uploaded anything relevant yet, answer from general "
    "knowledge and briefly note that you can answer more precisely once they upload a "
    "related document."
)

GROUNDED_SYSTEM_PROMPT = (
    "You are the assistant inside a Multimodal Document Intelligence app. Answer the user's "
    "question using ONLY the numbered context excerpts below, which were retrieved from "
    "documents they uploaded. Each excerpt is labeled with its source filename and page "
    "number. When you use information from an excerpt, mention which document and page it "
    "came from (e.g. \"according to invoice.pdf, page 2\"). If the excerpts don't contain "
    "the answer, say so plainly rather than guessing from general knowledge."
)


async def _call_openrouter(messages: list[dict], *, model_override: str | None = None) -> str:
    model = model_override or settings.openrouter_model
    async with httpx.AsyncClient(base_url=settings.openrouter_base_url, timeout=60.0) as client:
        response = await client.post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={"model": model, "messages": messages},
        )
        response.raise_for_status()
        data = response.json()

    # OpenRouter can return HTTP 200 with an `{"error": {...}}` body (e.g. upstream model
    # timeouts) instead of the usual `choices` payload — surface that as a clear 502 rather
    # than crashing on a KeyError.
    if "choices" not in data:
        message = data.get("error", {}).get("message", "Unknown error from OpenRouter")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"OpenRouter error: {message}")

    return data["choices"][0]["message"]["content"]


async def describe_visual_async(image_b64: str, hint: str = "") -> str:
    """Send a base64-encoded image to the vision model and return a search-friendly description."""
    prompt = "Describe this image in detail for document search and question-answering indexing."
    if hint:
        prompt += f" Context from document: {hint}"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ]}]
    return await _call_openrouter(messages, model_override=settings.openrouter_vision_model)


async def get_assistant_reply(history: list[dict[str, str]]) -> str:
    """Call OpenRouter's OpenAI-compatible chat-completions endpoint and return the reply text.

    `history` is the conversation so far as a chronological list of
    {"role": "user" | "assistant", "content": str} dicts.
    """
    return await _call_openrouter([{"role": "system", "content": SYSTEM_PROMPT}, *history])


async def get_grounded_reply(history: list[dict[str, str]], context_chunks: list[dict[str, str]]) -> str:
    """Like `get_assistant_reply`, but augments the prompt with retrieved document excerpts.

    `context_chunks` is a list of {"filename", "page_number", "text"} dicts, ordered most
    relevant first (as returned by the vector search).
    """
    excerpts = "\n\n".join(
        f"[{i}] Source: {chunk['filename']}, page {chunk['page_number']}\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks, start=1)
    )
    context_message = {"role": "system", "content": f"Context excerpts:\n\n{excerpts}"}
    messages = [{"role": "system", "content": GROUNDED_SYSTEM_PROMPT}, context_message, *history]
    return await _call_openrouter(messages)
