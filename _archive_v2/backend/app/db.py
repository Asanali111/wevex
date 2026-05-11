from supabase import create_client
from app.config import get_settings

_settings = get_settings()

# Service role client for agent API and background tasks
_service_client = None


def get_service_client():
    global _service_client
    if _service_client is None:
        _service_client = create_client(
            _settings.supabase_url,
            _settings.supabase_service_role_key
        )
    return _service_client


def get_user_client(jwt: str):
    """Create a Supabase client authenticated as the user. RLS policies apply."""
    client = create_client(
        _settings.supabase_url,
        _settings.supabase_anon_key
    )
    client.postgrest.auth(jwt)
    return client
