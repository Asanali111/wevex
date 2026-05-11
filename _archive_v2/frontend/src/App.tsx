import { useState, useEffect } from 'react'
import type { User } from '@supabase/supabase-js'
import { supabase, getUser } from './lib/supabase'
import { onOpenUrl } from '@tauri-apps/plugin-deep-link'
import Layout from './components/Layout'
import Auth from './components/Auth'
import Dashboard from './components/Dashboard'
import SkillList from './components/SkillList'
import SkillEditor from './components/SkillEditor'
import SkillView from './components/SkillView'
import SearchResults from './components/SearchResults'

export type Page =
  | { name: 'dashboard' }
  | { name: 'skills' }
  | { name: 'skill-new' }
  | { name: 'skill-view'; skillName: string }
  | { name: 'skill-edit'; skillName: string }
  | { name: 'search'; query: string }
  | { name: 'suggestions' }
  | { name: 'settings' }

function App() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState<Page>({ name: 'dashboard' })

  useEffect(() => {
    getUser().then(u => {
      setUser(u)
      setLoading(false)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })

    let unlistenDeepLink: (() => void) | undefined;
    const isTauri = '__TAURI__' in window;
    
    if (isTauri) {
      onOpenUrl((urls) => {
        const url = urls[0];
        if (url && url.includes('companybrain://auth/callback')) {
          const hashParams = url.split('#')[1];
          if (hashParams) {
            const params = new URLSearchParams(hashParams);
            const access_token = params.get('access_token');
            const refresh_token = params.get('refresh_token');
            if (access_token && refresh_token) {
              supabase.auth.setSession({ access_token, refresh_token });
            }
          }
        }
      }).then((unlisten) => {
        unlistenDeepLink = unlisten;
      }).catch(console.error);
    }

    return () => {
      subscription.unsubscribe();
      if (unlistenDeepLink) unlistenDeepLink();
    }
  }, [])

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: 'var(--bg)',
        color: 'var(--text-muted)',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.875rem'
      }}>
        initializing...
      </div>
    )
  }

  if (!user) {
    return <Auth onAuth={() => setUser(null)} />
  }

  const renderPage = () => {
    switch (page.name) {
      case 'dashboard':
        return <Dashboard onNavigate={setPage} />
      case 'skills':
        return <SkillList onNavigate={setPage} />
      case 'skill-new':
        return <SkillEditor onNavigate={setPage} />
      case 'skill-view':
        return <SkillView skillName={page.skillName} onNavigate={setPage} />
      case 'skill-edit':
        return <SkillEditor skillName={page.skillName} onNavigate={setPage} />
      case 'search':
        return <SearchResults query={page.query} onNavigate={setPage} />
      case 'suggestions':
        return <div style={{ padding: '2rem' }}><h2>Suggestions</h2><p style={{ color: 'var(--text-muted)', marginTop: '1rem' }}>Pattern Discovery Engine coming in Phase 2.</p></div>
      case 'settings':
        return <div style={{ padding: '2rem' }}><h2>Settings</h2><p style={{ color: 'var(--text-muted)', marginTop: '1rem' }}>Team settings and API keys coming soon.</p></div>
      default:
        return <Dashboard onNavigate={setPage} />
    }
  }

  return (
    <Layout user={user} page={page} onNavigate={setPage}>
      <div className="animate-fade-in" style={{ height: '100%' }}>
        {renderPage()}
      </div>
    </Layout>
  )
}

export default App
