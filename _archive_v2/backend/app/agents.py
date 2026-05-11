from fastapi import APIRouter, Depends, HTTPException, status
from app.db import get_service_client
from app.auth import get_agent_team
from app.models import AgentContextRequest, AgentContextResponse, SearchResult
from app.embeddings import generate_embedding

router = APIRouter(prefix="/agent", tags=["agents"])


async def get_embedding(text: str) -> list:
    """Generate an embedding via the configured provider."""
    return await generate_embedding(text)


@router.post("/context", response_model=AgentContextResponse)
async def get_agent_context(
    request: AgentContextRequest,
    team_id: str = Depends(get_agent_team)
):
    """
    Retrieve relevant skills for an agent based on intent.
    Authenticated via API key (not user JWT).
    """
    try:
        embedding = await get_embedding(request.intent)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding generation failed: {str(e)}"
        )
    
    service = get_service_client()
    
    try:
        result = service.rpc(
            "search_skills",
            {
                "query_embedding": embedding,
                "team_id_filter": team_id,
                "match_threshold": 0.6,
                "match_count": request.limit
            }
        ).execute()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search failed: {str(e)}"
        )
    
    skills = []
    for row in (result.data or []):
        skills.append(SearchResult(
            id=row.get("id", ""),
            name=row.get("name", ""),
            description=row.get("description", ""),
            content=row.get("content", ""),
            version=row.get("version", "1.0.0"),
            status=row.get("status", "active"),
            author=row.get("author", "unknown"),
            tags=row.get("tags", []),
            similarity=row.get("similarity", 0.0)
        ))
    
    return AgentContextResponse(skills=skills)


@router.get("/skills/{name}")
async def get_skill_for_agent(
    name: str,
    team_id: str = Depends(get_agent_team)
):
    """Get a specific skill by name for agent consumption."""
    service = get_service_client()
    result = service.table("skills").select("*").eq(
        "team_id", team_id
    ).eq("name", name).eq("status", "active").execute()
    
    if not result.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Skill not found")
    
    return result.data[0]
