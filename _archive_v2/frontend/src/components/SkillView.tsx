import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import type { Page } from '../App'

interface Skill {
  id: string
  name: string
  description: string
  content: string
  author: string
  version: string
  status: string
  confidence: number | null
  suggested_by: string | null
  source_threads: string[]
  approved_by: string | null
  approved_at: string | null
  last_validated: string | null
  requires_approval: boolean
  tags: string[]
  created_at: string
  updated_at: string
}

interface SkillViewProps {
  skillName: string
  onNavigate: (page: Page) => void
}

export default function SkillView({ skillName, onNavigate }: SkillViewProps) {
  const [skill, setSkill] = useState<Skill | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      const { data } = await supabase
        .from('skills')
        .select('*')
        .eq('name', skillName)
        .single()

      setSkill(data)
      setLoading(false)
    }
    load()
  }, [skillName])

  const handleDelete = async () => {
    if (!confirm(`Deprecate skill "${skillName}"?`)) return
    await supabase.from('skills').update({ status: 'deprecated' }).eq('name', skillName)
    onNavigate({ name: 'skills' })
  }

  if (loading) {
    return (
      <div style={{ padding: '2rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        loading...
      </div>
    )
  }

  if (!skill) {
    return (
      <div style={{ padding: '2rem' }}>
        <h2>Skill not found</h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          "{skillName}" does not exist in your library.
        </p>
        <button className="btn" style={{ marginTop: '1rem' }} onClick={() => onNavigate({ name: 'skills' })}>
          Back to Skills
        </button>
      </div>
    )
  }

  return (
    <div style={{ padding: '2rem', maxWidth: '800px' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        marginBottom: '1.5rem',
        gap: '1rem'
      }}>
        <div style={{ minWidth: 0 }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '0.5rem',
            flexWrap: 'wrap'
          }}>
            <h1 style={{ margin: 0 }}>{skill.name}</h1>
            <span className={`badge badge-${skill.status}`}>{skill.status}</span>
            {skill.requires_approval && (
              <span className="badge" style={{ background: 'rgba(239, 68, 68, 0.08)', color: '#EF4444', borderColor: 'rgba(239, 68, 68, 0.2)' }}>
                requires approval
              </span>
            )}
          </div>
          <p style={{ color: 'var(--text-tertiary)', fontSize: '0.875rem' }}>
            {skill.description}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
          <button
            className="btn btn-sm"
            onClick={() => onNavigate({ name: 'skill-edit', skillName: skill.name })}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
            </svg>
            Edit
          </button>
          <button
            className="btn btn-ghost btn-sm"
            onClick={handleDelete}
            style={{ color: 'var(--text-muted)' }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
          </button>
        </div>
      </div>

      {/* Metadata */}
      <div className="card" style={{
        padding: '1rem 1.25rem',
        marginBottom: '1.5rem',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
        gap: '1rem'
      }}>
        <MetaItem label="Version" value={`v${skill.version}`} />
        <MetaItem label="Author" value={skill.author} />
        {skill.approved_by && <MetaItem label="Approved by" value={skill.approved_by} />}
        {skill.approved_at && <MetaItem label="Approved" value={formatDate(skill.approved_at)} />}
        {skill.last_validated && <MetaItem label="Last validated" value={formatDate(skill.last_validated)} />}
        {skill.suggested_by && (
          <MetaItem
            label="Suggested by"
            value={skill.suggested_by}
            accent={`${Math.round((skill.confidence || 0) * 100)}% confidence`}
          />
        )}
      </div>

      {/* Tags */}
      {skill.tags.length > 0 && (
        <div style={{ display: 'flex', gap: '0.375rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          {skill.tags.map(tag => (
            <span key={tag} className="tag">{tag}</span>
          ))}
        </div>
      )}

      {/* Content */}
      <div
        className="markdown-content"
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          padding: '1.5rem'
        }}
        dangerouslySetInnerHTML={{ __html: renderMarkdown(skill.content) }}
      />

      {/* Source threads */}
      {skill.source_threads && skill.source_threads.length > 0 && (
        <div style={{ marginTop: '1.5rem' }}>
          <h4 style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>
            Source Threads
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            {skill.source_threads.map((thread, i) => (
              <code key={i} style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{thread}</code>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function MetaItem({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div>
      <div style={{
        fontSize: '0.625rem',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        color: 'var(--text-muted)',
        fontWeight: 600,
        marginBottom: '0.25rem'
      }}>
        {label}
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '0.8125rem',
        color: 'var(--text-secondary)'
      }}>
        {value}
      </div>
      {accent && (
        <div style={{
          fontSize: '0.6875rem',
          color: 'var(--accent)',
          marginTop: '0.125rem'
        }}>
          {accent}
        </div>
      )}
    </div>
  )
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
}

function renderMarkdown(md: string): string {
  return md
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^\*\*(.*)\*\*/gim, '<strong>$1</strong>')
    .replace(/^\* \*\*(.*?)\*\*: (.*$)/gim, '<li><strong>$1</strong>: $2</li>')
    .replace(/^\d+\. \*\*(.*?)\*\*: (.*$)/gim, '<li><strong>$1</strong>: $2</li>')
    .replace(/^- (.*$)/gim, '<li>$1</li>')
    .replace(/`([^`]+)`/gim, '<code>$1</code>')
    .replace(/\n/gim, '<br>')
}
