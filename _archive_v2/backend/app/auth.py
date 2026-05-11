import hashlib
import secrets
from fastapi import Depends, HTTPException, Header, status, Request
from jose import jwt, JWTError
from app.config import get_settings
from app.db import get_service_client

settings = get_settings()


def generate_api_key() -> str:
    """Generate a random API key prefixed with 'brain_'."""
    return f"brain_{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    """Hash an API key using SHA-256. For MVP; upgrade to bcrypt in production."""
    return hashlib.sha256(key.encode()).hexdigest()


async def get_current_user(request: Request):
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header"
        )
    token = auth[7:]
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"]
        )
        user_id = payload.get("sub")
        if not user_id:
            raise JWTError()
        return {
            "id": user_id,
            "email": payload.get("email", ""),
            "token": token
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


async def get_user_team(user: dict = Depends(get_current_user)):
    """Get the user's primary team. Assumes one team per user for MVP."""
    client = get_service_client()
    result = client.table("team_members").select("team_id, role").eq(
        "user_id", user["id"]
    ).execute()
    
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not a member of any team"
        )
    
    membership = result.data[0]
    return {
        "team_id": membership["team_id"],
        "role": membership["role"],
        "user_id": user["id"],
        "token": user["token"]
    }


async def get_agent_team(
    x_api_key: str = Header(None, alias=settings.agent_api_key_header)
):
    """Authenticate agent requests via API key."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {settings.agent_api_key_header} header"
        )
    
    key_hash = hash_api_key(x_api_key)
    client = get_service_client()
    
    result = client.table("api_keys").select("team_id, name").eq(
        "key_hash", key_hash
    ).execute()
    
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    key_record = result.data[0]
    
    # Update last_used_at
    client.table("api_keys").update({
        "last_used_at": "now()"
    }).eq("key_hash", key_hash).execute()
    
    return key_record["team_id"]
