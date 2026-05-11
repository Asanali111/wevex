from fastapi import APIRouter, Depends, HTTPException, status
from app.db import get_user_client, get_service_client
from app.auth import get_user_team
from app.models import SkillCreate, SkillUpdate, SkillOut, SearchQuery, SearchResult
from app.embeddings import generate_embedding

router = APIRouter(prefix="/skills", tags=["skills"])


async def get_embedding(text: str) -> list:
    """Generate an embedding via the configured provider."""
    return await generate_embedding(text)


@router.get("", response_model=list[SkillOut])
async def list_skills(
    team: dict = Depends(get_user_team),
    status: str = None,
    tag: str = None
):
    """List all skills for the user's team."""
    supabase = get_user_client(team["token"])
    query = supabase.table("skills").select("*")
    
    if status:
        query = query.eq("status", status)
    if tag:
        query = query.contains("tags", [tag])
    
    result = query.order("name").execute()
    return result.data or []


@router.get("/{name}", response_model=SkillOut)
async def get_skill(name: str, team: dict = Depends(get_user_team)):
    """Get a specific skill by name."""
    supabase = get_user_client(team["token"])
    result = supabase.table("skills").select("*").eq(
        "team_id", team["team_id"]
    ).eq("name", name).execute()
    
    if not result.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Skill not found")
    
    return result.data[0]


@router.post("", response_model=SkillOut, status_code=status.HTTP_201_CREATED)
async def create_skill(skill: SkillCreate, team: dict = Depends(get_user_team)):
    """Create a new skill with auto-generated embedding."""
    # Check for duplicate name
    service = get_service_client()
    existing = service.table("skills").select("id").eq(
        "team_id", team["team_id"]
    ).eq("name", skill.name).execute()
    
    if existing.data:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Skill '{skill.name}' already exists"
        )
    
    # Generate embedding from description + first 500 chars of content
    embedding_text = f"{skill.name}\n{skill.description}\n{skill.content[:500]}"
    embedding = await get_embedding(embedding_text)
    
    data = skill.model_dump()
    data["team_id"] = team["team_id"]
    data["embedding"] = embedding
    data["author"] = team.get("user_id", data.get("author", "unknown"))
    
    supabase = get_user_client(team["token"])
    result = supabase.table("skills").insert(data).execute()
    
    return result.data[0]


@router.put("/{name}", response_model=SkillOut)
async def update_skill(name: str, update: SkillUpdate, team: dict = Depends(get_user_team)):
    """Update a skill. Regenerates embedding if content or description changes."""
    supabase = get_user_client(team["token"])
    
    # Get existing skill
    existing = supabase.table("skills").select("*").eq(
        "team_id", team["team_id"]
    ).eq("name", name).execute()
    
    if not existing.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Skill not found")
    
    skill = existing.data[0]
    update_data = {k: v for k, v in update.model_dump().items() if v is not None}
    
    # Regenerate embedding if content changed
    if "content" in update_data or "description" in update_data:
        content = update_data.get("content", skill["content"])
        description = update_data.get("description", skill["description"])
        embedding_text = f"{name}\n{description}\n{content[:500]}"
        update_data["embedding"] = await get_embedding(embedding_text)
    
    # Auto-increment version on content change
    if "content" in update_data or "description" in update_data:
        parts = skill["version"].split(".")
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
            update_data["version"] = ".".join(parts)
    
    result = supabase.table("skills").update(update_data).eq(
        "team_id", team["team_id"]
    ).eq("name", name).execute()
    
    return result.data[0]


@router.delete("/{name}")
async def delete_skill(name: str, team: dict = Depends(get_user_team)):
    """Soft-delete a skill by setting status to deprecated."""
    supabase = get_user_client(team["token"])
    
    result = supabase.table("skills").update({
        "status": "deprecated",
        "updated_at": "now()"
    }).eq("team_id", team["team_id"]).eq("name", name).execute()
    
    if not result.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Skill not found")
    
    return {"message": f"Skill '{name}' deprecated"}


@router.post("/search", response_model=list[SearchResult])
async def search_skills(query: SearchQuery, team: dict = Depends(get_user_team)):
    """Semantic search across skills using pgvector."""
    embedding = await get_embedding(query.query)
    
    # Use Supabase RPC for vector search
    service = get_service_client()
    result = service.rpc(
        "search_skills",
        {
            "query_embedding": embedding,
            "team_id_filter": str(team["team_id"]),
            "match_threshold": query.threshold,
            "match_count": query.limit
        }
    ).execute()
    
    return result.data or []
