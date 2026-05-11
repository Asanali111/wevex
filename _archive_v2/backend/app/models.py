from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class SkillBase(BaseModel):
    name: str = Field(..., pattern=r"^[a-z0-9-]+$")
    description: str = Field(..., max_length=200)
    content: str
    author: str
    version: str = "1.0.0"
    status: str = "draft"  # draft | active | deprecated | pending_review
    confidence: Optional[float] = None
    suggested_by: Optional[str] = None
    source_threads: List[str] = Field(default_factory=list)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    last_validated: Optional[datetime] = None
    requires_approval: bool = False
    tags: List[str] = Field(default_factory=list)


class SkillCreate(SkillBase):
    pass


class SkillUpdate(BaseModel):
    description: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = None
    version: Optional[str] = None
    status: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    last_validated: Optional[datetime] = None
    requires_approval: Optional[bool] = None
    tags: Optional[List[str]] = None


class SkillOut(SkillBase):
    id: str
    team_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    threshold: float = 0.7
    limit: int = Field(10, ge=1, le=50)


class SearchResult(BaseModel):
    id: str
    name: str
    description: str
    content: str
    version: str
    status: str
    author: str
    tags: List[str]
    similarity: float


class AgentContextRequest(BaseModel):
    intent: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(3, ge=1, le=10)


class AgentContextResponse(BaseModel):
    skills: List[SearchResult]


class SuggestionOut(BaseModel):
    id: str
    team_id: str
    skill_name: Optional[str]
    suggestion_type: str
    reason: str
    draft_content: Optional[str]
    source_activities: List[str]
    confidence: Optional[float]
    status: str
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]
    created_at: datetime
