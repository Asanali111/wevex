import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import type { Page } from '../App'

interface SkillCount {
  status: string
  count: number
}

interface DashboardProps {
  onNavigate: (page: Page) => void
}

export default function Dashboard({ onNavigate }: DashboardProps) {
  const [counts, setCounts] = useState<SkillCount[]>([])
  const [recentSkills, setRecentSkills] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      const { data: skills } = await supabase.from('skills').select('*').order('updated_at', { ascending: false }).limit(5)
      setRecentSkills(skills || [])

      const { data: countsData } = await supabase.from('skills').select('status')
      const statusMap = new Map<string, number>()
      countsData?.forEach(s => {
        statusMap.set(s.status, (statusMap.get(s.status) || 0) + 1)
      })
      setCounts(Array.from(statusMap.entries()).map(([status, count]) => ({ status, count })))
      setLoading(false)
    }
    load()
  }, [])

  const total = counts.reduce((a, b) => a + b.count, 0)
  const active = counts.find(c => c.status === 'active')?.count || 0
  const draft = counts.find(c => c.status === 'draft')?.count || 0

  return (
    <div style={{ padding: '2rem', maxWidth: '960px' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ marginBottom: '0.25rem' }}>Dashboard</h1>
        <p style={{ color: 'var(--text-tertiary)', fontSize: '0.875rem' }}>
          Your team's knowledge layer overview.
        </p>
      </div>

      {/* Stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '1rem',
        marginBottom: '2.5rem'
      }}>
        <StatCard label="Total Skills" value={total} accent={false} />
        <StatCard label="Active" value={active} accent />
        <StatCard label="Draft" value={draft} accent={false} />
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '2.5rem' }}>
        <button
          className="btn btn-primary"
          onClick={() => onNavigate({ name: 'skill-new' })}
        >
          <PlusIcon />
          New Skill
        </button>
        <button
          className="btn"
          onClick={() => onNavigate({ name: 'search', query: '' })}
        >
          <SearchIcon />
          Search
        </button>
        <button
          className="btn btn-ghost"
          onClick={() => onNavigate({ name: 'skills' })}
        >
          View All
        </button>
      </div>

      {/* Recent Skills */}
      <div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '1rem'
        }}>
          <h3>Recent Skills</h3>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => onNavigate({ name: 'skills' })}
          >
            View all →
          </button>
        </div>

        {loading ? (
          <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.8125rem' }}>
            loading...
          </div>
        ) : recentSkills.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📋</div>
            <div style={{ fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
              No skills yet
            </div>
            <div style={{ fontSize: '0.8125rem' }}>
              Create your first skill to get started.
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {recentSkills.map(skill => (
              <button
                key={skill.id}
                onClick={() => onNavigate({ name: 'skill-view', skillName: skill.name })}
                className="card card-hover"
                style={{
                  padding: '1rem',
                  textAlign: 'left',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '1rem'
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <div style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.8125rem',
                    color: 'var(--text-primary)',
                    marginBottom: '0.25rem',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    {skill.name}
                  </div>
                  <div style={{
                    fontSize: '0.75rem',
                    color: 'var(--text-tertiary)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    {skill.description}
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
                  <span className={`badge badge-${skill.status}`}>{skill.status}</span>
                  <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    v{skill.version}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({ label, value, accent }: { label: string; value: number; accent: boolean }) {
  return (
    <div className="card" style={{
      padding: '1.25rem',
      borderColor: accent ? 'var(--accent-dark)' : undefined
    }}>
      <div style={{
        fontSize: '1.75rem',
        fontWeight: 700,
        color: accent ? 'var(--accent)' : 'var(--text-primary)',
        fontFamily: 'var(--font-mono)',
        letterSpacing: '-0.04em',
        marginBottom: '0.375rem'
      }}>
        {value}
      </div>
      <div style={{
        fontSize: '0.75rem',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
        color: 'var(--text-tertiary)',
        fontWeight: 500
      }}>
        {label}
      </div>
    </div>
  )
}

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  )
}

function SearchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  )
}
