import type { ReactNode } from 'react'
import type { User } from '@supabase/supabase-js'
import type { Page } from '../App'
import Sidebar from './Sidebar'

interface LayoutProps {
  children: ReactNode
  user: User
  page: Page
  onNavigate: (page: Page) => void
}

export default function Layout({ children, user, page, onNavigate }: LayoutProps) {
  return (
    <div style={{
      display: 'flex',
      height: '100vh',
      overflow: 'hidden',
      background: 'var(--bg)'
    }}>
      <Sidebar user={user} page={page} onNavigate={onNavigate} />
      <main style={{
        flex: 1,
        overflow: 'auto',
        minWidth: 0
      }}>
        {children}
      </main>
    </div>
  )
}
