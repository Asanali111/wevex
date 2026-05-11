import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import type { Page } from '../App'

interface Skill {
  id: string
  name: string
  description: string
  status: string
  version: string
  author: string
  tags: string[]
  updated_at: string
}

interface SkillListProps {
  onNavigate: (page: Page) => void
}

const FILTERS = ['all', 'active', 'draft', 'deprecated'] as const

export default function SkillList({ onNavigate }: SkillListProps) {
  const [skills, setSkills] = useState<Skill[]>([])
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>('all')
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      let query = supabase.from('skills').select('*').order('name')
      if (filter !== 'all') query = query.eq('status', filter)
      const { data } = await query
      setSkills(data || [])
      setLoading(false)
    }
    load()
  }, [filter])

  const filteredSkills = skills.filter(s =>
    search === '' ||
    s.name.toLowerCase().includes(search.toLowerCase()) ||
    s.description.toLowerCase().includes(search.toLowerCase()) ||
    s.tags.some(t => t.toLowerCase().includes(search.toLowerCase()))
  )

  return (
    <div style={{ padding: '2rem', maxWidth: '960px' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '2rem'
      }}>
        <div>
          <h1 style={{ marginBottom: '0.25rem' }}>Skills</h1>
          <p style={{ color: 'var(--text-tertiary)', fontSize: '0.875rem' }}>
            {skills.length} procedure{skills.length !== 1 ? 's' : ''} in your library
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => onNavigate({ name: 'skill-new' })}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Skill
        </button>
      </div>

      {/* Search + Filters */}
      <div style={{
        display: 'flex',
        gap: '1rem',
        marginBottom: '1.5rem',
        alignItems: 'center'
      }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" style={{
            position: 'absolute',
            left: '0.75rem',
            top: '50%',
            transform: 'translateY(-50%)'
          }}>
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Filter skills..."
            style={{ paddingLeft: '2.25rem' }}
          />
        </div>

        <div style={{ display: 'flex', gap: '0.25rem' }}>
          {FILTERS.map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className="btn btn-sm"
              style={{
                background: filter === f ? 'var(--accent)' : undefined,
                borderColor: filter === f ? 'var(--accent)' : undefined,
                color: filter === f ? '#fff' : undefined
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Skills Grid */}
      {loading ? (
        <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          loading...
        </div>
      ) : filteredSkills.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">📋</div>
          <div style={{ fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
            No skills found
          </div>
          <div style={{ fontSize: '0.8125rem' }}>
            {search ? 'Try a different search term.' : 'Create your first skill.'}
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {filteredSkills.map(skill => (
            <button
              key={skill.id}
              onClick={() => onNavigate({ name: 'skill-view', skillName: skill.name })}
              className="card card-hover"
              style={{
                padding: '1rem 1.25rem',
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '1rem',
                border: '1px solid var(--border)'
              }}
            >
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  marginBottom: '0.375rem'
                }}>
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.8125rem',
                    color: 'var(--text-primary)',
                    fontWeight: 500
                  }}>
                    {skill.name}
                  </span>
                  <span className={`badge badge-${skill.status}`}>
                    {skill.status}
                  </span>
                </div>
                <div style={{
                  fontSize: '0.8125rem',
                  color: 'var(--text-tertiary)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  marginBottom: '0.375rem'
                }}>
                  {skill.description}
                </div>
                <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
                  {skill.tags.map(tag => (
                    <span key={tag} className="tag">{tag}</span>
                  ))}
                </div>
              </div>

              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                flexShrink: 0
              }}>
                <span style={{
                  fontSize: '0.6875rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)'
                }}>
                  v{skill.version}
                </span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2">
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
