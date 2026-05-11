from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Company Brain API"
    debug: bool = False

    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""

    # Embedding provider: openai | gemini | fireworks
    embedding_provider: str = "openai"
    embedding_dimension: int = 1536

    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    # Gemini (Google)
    gemini_api_key: str = ""
    gemini_embedding_model: str = "models/text-embedding-004"

    # Fireworks.ai (OpenAI-compatible embedding endpoint)
    fireworks_api_key: str = ""
    fireworks_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"

    agent_api_key_header: str = "x-api-key"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
