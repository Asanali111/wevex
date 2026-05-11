-- ============================================================
-- Company Brain - Supabase Schema
-- Run this in the Supabase SQL Editor
-- ============================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- NOTE: VECTOR(1536) is the default dimension for OpenAI embeddings.
-- If you use Gemini (768-dim) or Fireworks (768-dim), the backend
-- zero-pads smaller embeddings to 1536. Cosine similarity is preserved
-- under zero-padding. If you switch to a >1536-dim model, update this
-- column and the EMBEDDING_DIMENSION env var accordingly.

-- ============================================================
-- TEAMS
-- ============================================================
CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- TEAM MEMBERS (links auth.users to teams)
-- ============================================================
CREATE TABLE team_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member')),
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(team_id, user_id)
);

-- ============================================================
-- SKILLS
-- ============================================================
CREATE TABLE skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    content TEXT NOT NULL,
    author TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0.0',
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'deprecated', 'pending_review')),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    suggested_by TEXT,
    source_threads JSONB DEFAULT '[]'::jsonb,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    last_validated TIMESTAMPTZ,
    requires_approval BOOLEAN DEFAULT false,
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(team_id, name)
);

-- Index for team-based lookups
CREATE INDEX idx_skills_team ON skills(team_id);
CREATE INDEX idx_skills_status ON skills(status);
CREATE INDEX idx_skills_tags ON skills USING gin(tags);

-- ============================================================
-- ACTIVITIES (for Pattern Discovery Engine)
-- ============================================================
CREATE TABLE activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    source TEXT NOT NULL,
    type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(team_id, source, external_id)
);

CREATE INDEX idx_activities_team ON activities(team_id);
CREATE INDEX idx_activities_source ON activities(source, type);

-- ============================================================
-- SUGGESTIONS
-- ============================================================
CREATE TABLE suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    skill_name TEXT,
    suggestion_type TEXT NOT NULL CHECK (suggestion_type IN ('new_skill', 'update', 'stale')),
    reason TEXT NOT NULL,
    draft_content TEXT,
    source_activities JSONB DEFAULT '[]'::jsonb,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_suggestions_team ON suggestions(team_id);
CREATE INDEX idx_suggestions_status ON suggestions(status);

-- ============================================================
-- API KEYS (for agent authentication)
-- ============================================================
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_team ON api_keys(team_id);

-- ============================================================
-- VECTOR SEARCH FUNCTION
-- ============================================================
CREATE OR REPLACE FUNCTION search_skills(
    query_embedding vector(1536),
    team_id_filter UUID,
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE(
    id UUID,
    name TEXT,
    description TEXT,
    content TEXT,
    version TEXT,
    status TEXT,
    author TEXT,
    tags TEXT[],
    similarity FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.id,
        s.name,
        s.description,
        s.content,
        s.version,
        s.status,
        s.author,
        s.tags,
        1 - (s.embedding <=> query_embedding) AS similarity
    FROM skills s
    WHERE s.team_id = team_id_filter
      AND s.status = 'active'
      AND s.embedding IS NOT NULL
      AND 1 - (s.embedding <=> query_embedding) > match_threshold
    ORDER BY s.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE suggestions ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Teams: members can view their teams
CREATE POLICY teams_isolation ON teams
    FOR ALL
    USING (
        id IN (
            SELECT team_id FROM team_members WHERE user_id = auth.uid()
        )
    );

-- Team members: view team memberships
CREATE POLICY team_members_isolation ON team_members
    FOR ALL
    USING (
        team_id IN (
            SELECT team_id FROM team_members WHERE user_id = auth.uid()
        )
    );

-- Skills: isolated by team membership
CREATE POLICY skills_isolation ON skills
    FOR ALL
    USING (
        team_id IN (
            SELECT team_id FROM team_members WHERE user_id = auth.uid()
        )
    );

-- Activities: isolated by team membership
CREATE POLICY activities_isolation ON activities
    FOR ALL
    USING (
        team_id IN (
            SELECT team_id FROM team_members WHERE user_id = auth.uid()
        )
    );

-- Suggestions: isolated by team membership
CREATE POLICY suggestions_isolation ON suggestions
    FOR ALL
    USING (
        team_id IN (
            SELECT team_id FROM team_members WHERE user_id = auth.uid()
        )
    );

-- API Keys: only admins can manage, but all team members can view (for reference)
CREATE POLICY api_keys_admin ON api_keys
    FOR ALL
    USING (
        team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = auth.uid() AND role = 'admin'
        )
    );

-- ============================================================
-- TRIGGERS
-- ============================================================

-- Auto-update updated_at on skills
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_skills_updated_at
    BEFORE UPDATE ON skills
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- SEED DATA (for testing)
-- ============================================================

-- Create a default team (optional, for testing)
-- INSERT INTO teams (name, slug) VALUES ('Engineering', 'engineering');
