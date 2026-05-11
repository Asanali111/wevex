"""
Multi-provider embedding abstraction.

Supports: openai, gemini, fireworks
Handles dimension mismatch by zero-padding (preserves cosine similarity).
"""

import httpx
import openai
from app.config import get_settings

settings = get_settings()


def _pad_embedding(embedding: list[float], target_dim: int) -> list[float]:
    """
    Pad an embedding with zeros to reach target dimension.

    Zero-padding preserves cosine similarity because:
    - dot(a + zeros, b + zeros) = dot(a, b)
    - |a + zeros| = |a|
    - So cos_sim stays identical.
    """
    current = len(embedding)
    if current >= target_dim:
        return embedding[:target_dim]
    return embedding + [0.0] * (target_dim - current)


async def _embed_openai(text: str) -> list[float]:
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text[:8192],
    )
    return response.data[0].embedding


async def _embed_gemini(text: str) -> list[float]:
    """Gemini embedding via REST (no google-generativeai dep needed)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{settings.gemini_embedding_model}:embedContent"
        f"?key={settings.gemini_api_key}"
    )
    payload = {"content": {"parts": [{"text": text[:8192]}]}}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    embedding = data["embedding"]["values"]
    return embedding


async def _embed_fireworks(text: str) -> list[float]:
    """Fireworks.ai via OpenAI-compatible endpoint."""
    client = openai.AsyncOpenAI(
        base_url=settings.fireworks_base_url,
        api_key=settings.fireworks_api_key,
    )
    response = await client.embeddings.create(
        model=settings.fireworks_embedding_model,
        input=text[:8192],
    )
    return response.data[0].embedding


async def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector using the configured provider.

    Raises ValueError if provider is unknown or required API key is missing.
    """
    provider = (settings.embedding_provider or "openai").lower().strip()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for embedding_provider=openai")
        embedding = await _embed_openai(text)

    elif provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for embedding_provider=gemini")
        embedding = await _embed_gemini(text)

    elif provider == "fireworks":
        if not settings.fireworks_api_key:
            raise ValueError("FIREWORKS_API_KEY is required for embedding_provider=fireworks")
        embedding = await _embed_fireworks(text)

    else:
        raise ValueError(
            f"Unknown embedding_provider: '{provider}'. "
            f"Supported: openai, gemini, fireworks"
        )

    return _pad_embedding(embedding, settings.embedding_dimension)
