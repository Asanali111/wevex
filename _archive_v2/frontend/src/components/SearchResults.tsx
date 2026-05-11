import { useState } from 'react'
import { apiFetch } from '../lib/supabase'
import type { Page } from '../App'

interface SearchResult {
  id: string
  name: string
  description: string
  content: string
  version: string
  status: string
  author: string
  tags: string[]
  similarity: number
}

interface SearchResultsProps {
  query: string
  onNavigate: (page: Page) => void
}

export default function SearchResults({ query: initialQuery, onNavigate }: SearchResultsProps) {
  const [query, setQuery] = useState(initialQuery)
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setSearched(true)

    try {
      const res = await apiFetch('/skills/search', {
        method: 'POST',
        body: JSON.stringify({ query, threshold: 0.6, limit: 20 })
      })

      if (res.ok) {
        const data = await res.json()
        setResults(data)
      } else {
        setResults([])
      }
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  // Auto-search if initial query provided
  useEffect(() => {
    if (initialQuery) {
      setQuery(initialQuery)
      handleSearch()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div style={{ padding: '2rem', maxWidth: '800px' }}>
      <h1 style={{ marginBottom: '1.5rem' }}>Search</h1>

      <form onSubmit={handleSearch} style={{ marginBottom: '2rem' }}>
        <div style={{
          display: 'flex',
          gap: '0.75rem'
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
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Describe what you're looking for..."
              style={{ paddingLeft: '2.25rem', fontSize: '0.9375rem' }}
            />
          </div>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? '...' : 'Search'}
          </button>
        </div>
      </form>

      {/* Results */}
      {searched && !loading && results.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-icon">🔍</div>
          <div style={{ fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
            No results found
          </div>
          <div style={{ fontSize: '0.8125rem' }}>
            Try a different description of what you need.
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div style={{
            fontSize: '0.75rem',
            color: 'var(--text-muted)',
            marginBottom: '0.5rem',
            fontFamily: 'var(--font-mono)'
          }}>
            {results.length} result{results.length !== 1 ? 's' : ''}
          </div>

          {results.map(result => (
            <button
              key={result.id}
              onClick={() => onNavigate({ name: 'skill-view', skillName: result.name })}
              className="card card-hover"
              style={{
                padding: '1.25rem',
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                gap: '1rem'
              }}
            >
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  marginBottom: '0.5rem'
                }}>
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.875rem',
                    color: 'var(--text-primary)',
                    fontWeight: 500
                  }}>
                    {result.name}
                  </span>
                  <span className={`badge badge-${result.status}`}>
                    {result.status}
                  </span>
                </div>
                <div style={{
                  fontSize: '0.8125rem',
                  color: 'var(--text-tertiary)',
                  marginBottom: '0.5rem'
                }}>
                  {result.description}
                </div>
                <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
                  {result.tags?.map(tag => (
                    <span key={tag} className="tag">{tag}</span>
                  ))}
                </div>
              </div>

              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-end',
                gap: '0.25rem',
                flexShrink: 0
              }}>
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  color: 'var(--accent)'
                }}>
                  {Math.round(result.similarity * 100)}%
                </div>
                <div style={{
                  fontSize: '0.625rem',
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em'
                }}>
                  match
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
