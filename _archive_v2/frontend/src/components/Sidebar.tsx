import type { User } from '@supabase/supabase-js'
import type { Page } from '../App'
import { signOut } from '../lib/supabase'

interface SidebarProps {
  user: User
  page: Page
  onNavigate: (page: Page) => void
}

const NAV_ITEMS: { label: string; page: Page; id: string }[] = [
  { label: 'Dashboard', page: { name: 'dashboard' }, id: 'dashboard' },
  { label: 'Skills', page: { name: 'skills' }, id: 'skills' },
  { label: 'Search', page: { name: 'search', query: '' }, id: 'search' },
  { label: 'Suggestions', page: { name: 'suggestions' }, id: 'suggestions' },
  { label: 'Settings', page: { name: 'settings' }, id: 'settings' },
]

function isActive(page: Page, id: string): boolean {
  return page.name === id
}

export default function Sidebar({ user, page, onNavigate }: SidebarProps) {
  return (
    <aside style={{
      width: '220px',
      background: 'var(--bg-surface)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0
    }}>
      {/* Logo */}
      <div style={{
        padding: '1.5rem 1.25rem',
        borderBottom: '1px solid var(--border)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            background: 'var(--accent)',
            borderRadius: '50%'
          }} />
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.875rem',
            fontWeight: 600,
            color: 'var(--text-primary)',
            letterSpacing: '-0.02em'
          }}>
            company brain
          </span>
        </div>
        <div style={{
          fontSize: '0.6875rem',
          color: 'var(--text-muted)',
          marginTop: '0.25rem',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '0.02em'
        }}>
          v1.0.0
        </div>
      </div>

      {/* Navigation */}
      <nav style={{
        flex: 1,
        padding: '0.75rem 0',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.125rem'
      }}>
        {NAV_ITEMS.map(item => {
          const active = isActive(page, item.id)
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.page)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.5rem 1.25rem',
                width: '100%',
                border: 'none',
                borderLeft: active ? '2px solid var(--accent)' : '2px solid transparent',
                background: active ? 'var(--accent-glow)' : 'transparent',
                color: active ? 'var(--accent)' : 'var(--text-secondary)',
                fontSize: '0.8125rem',
                fontWeight: 500,
                fontFamily: 'var(--font-sans)',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.12s ease'
              }}
              onMouseEnter={e => {
                if (!active) {
                  e.currentTarget.style.background = 'var(--bg-hover)'
                  e.currentTarget.style.color = 'var(--text-primary)'
                }
              }}
              onMouseLeave={e => {
                if (!active) {
                  e.currentTarget.style.background = 'transparent'
                  e.currentTarget.style.color = 'var(--text-secondary)'
                }
              }}
            >
              <NavIcon id={item.id} active={active} />
              {item.label}
            </button>
          )
        })}
      </nav>

      {/* User */}
      <div style={{
        padding: '0.75rem 1.25rem',
        borderTop: '1px solid var(--border)'
      }}>
        <div style={{
          fontSize: '0.75rem',
          color: 'var(--text-tertiary)',
          marginBottom: '0.5rem',
          fontFamily: 'var(--font-mono)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          {user.email}
        </div>
        <button
          onClick={() => signOut()}
          className="btn btn-ghost btn-sm"
          style={{ width: '100%', justifyContent: 'center' }}
        >
          Sign out
        </button>
      </div>
    </aside>
  )
}

function NavIcon({ id, active }: { id: string; active: boolean }) {
  const color = active ? 'var(--accent)' : 'currentColor'
  const icons: Record<string, JSX.Element> = {
    dashboard: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
      </svg>
    ),
    skills: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <line x1="10" y1="9" x2="8" y2="9" />
      </svg>
    ),
    search: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    ),
    suggestions: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="5" x2="12" y2="19" />
        <line x1="5" y1="12" x2="19" y2="12" />
      </svg>
    ),
    settings: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
  }
  return icons[id] || null
}
