import { useEffect, useState } from 'react'
import { BookOpen, ChevronDown, Command, MessageSquare, Plus, Settings2 } from 'lucide-react'
import { ChatPage } from './pages/ChatPage'
import { DocumentsPage } from './pages/DocumentsPage'
import { createWorkspace, listWorkspaces } from './services/workspaces'
import type { Workspace } from './types/api'

type Page = 'chat' | 'documents'
const defaultUser = import.meta.env.VITE_DEFAULT_USER_ID || 'demo-user'
const configuredWorkspace = Number(import.meta.env.VITE_DEFAULT_WORKSPACE_ID || 0)

export default function App() {
  const [page, setPage] = useState<Page>('chat')
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [workspaceId, setWorkspaceId] = useState<number | null>(() => Number(localStorage.getItem('nodagent.workspace')) || configuredWorkspace || null)
  const [userId, setUserId] = useState(() => localStorage.getItem('nodagent.user') || defaultUser)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [newWorkspaceName, setNewWorkspaceName] = useState('')

  useEffect(() => {
    listWorkspaces().then(items => {
      setWorkspaces(items)
      setWorkspaceId(current => items.find(item => item.id === current)?.id ?? items[0]?.id ?? null)
      setError('')
    }).catch(err => setError(err instanceof Error ? err.message : String(err))).finally(() => setLoading(false))
  }, [])
  useEffect(() => { if (workspaceId) localStorage.setItem('nodagent.workspace', String(workspaceId)) }, [workspaceId])
  useEffect(() => { localStorage.setItem('nodagent.user', userId) }, [userId])

  const workspace = workspaces.find(item => item.id === workspaceId)
  async function addWorkspace() {
    if (!newWorkspaceName.trim()) return
    try {
      const created = await createWorkspace(newWorkspaceName.trim())
      setWorkspaces(items => [...items, created])
      setWorkspaceId(created.id)
      setNewWorkspaceName('')
      setError('')
    } catch (err) { setError(err instanceof Error ? err.message : String(err)) }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Command size={20} strokeWidth={2.5} /></span><span>NodAgent<small>AI KNOWLEDGE WORKSPACE</small></span></div>
      <div className="sidebar-section-label">WORKSPACE</div>
      <button className="workspace-switch" onClick={() => setSettingsOpen(value => !value)} aria-expanded={settingsOpen}>
        <span className="workspace-avatar">{workspace?.name?.slice(0, 1).toUpperCase() || 'N'}</span>
        <span className="workspace-copy"><strong>{workspace?.name || (loading ? 'Loading...' : 'Select workspace')}</strong><small>{workspace ? `Workspace #${workspace.id}` : 'Get started'}</small></span>
        <ChevronDown size={15} />
      </button>
      {settingsOpen && <div className="workspace-panel">
        <label htmlFor="workspace-select">Current workspace</label>
        <select id="workspace-select" value={workspaceId ?? ''} onChange={event => setWorkspaceId(Number(event.target.value))}>
          {workspaces.map(item => <option key={item.id} value={item.id}>{item.name} (#{item.id})</option>)}
        </select>
        <label htmlFor="new-workspace">New workspace</label>
        <div className="inline-form"><input id="new-workspace" placeholder="Workspace name" value={newWorkspaceName} onChange={event => setNewWorkspaceName(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') void addWorkspace() }} /><button onClick={() => void addWorkspace()} aria-label="Create workspace"><Plus size={16} /></button></div>
        <label htmlFor="user-id">User ID</label>
        <input id="user-id" value={userId} onChange={event => setUserId(event.target.value)} maxLength={100} />
      </div>}
      <div className="sidebar-section-label navigation-label">NAVIGATION</div>
      <nav className="nav-list" aria-label="Main navigation">
        <button className={page === 'chat' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('chat')}><MessageSquare size={18} />Chat<span className="nav-hint">⌘ 1</span></button>
        <button className={page === 'documents' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('documents')}><BookOpen size={18} />Documents</button>
      </nav>
      <div className="sidebar-bottom"><div className="sidebar-info"><span className="status-dot" /> LangGraph · Hybrid RAG · MCP</div><button className="sidebar-settings" onClick={() => setSettingsOpen(value => !value)}><Settings2 size={17} /> Workspace settings</button></div>
    </aside>
    <div className="main-area">
      <header className="topbar"><div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-slash">/</span><strong>{page === 'chat' ? 'Chat' : 'Documents'}</strong></div><div className="topbar-right"><span className="workspace-pill"><span className="status-dot" />{workspace?.name || 'No workspace'}</span><span className="user-pill">{userId || 'No user'}</span></div></header>
      {error && <div className="global-error" role="alert">{error}<button onClick={() => setError('')}>Dismiss</button></div>}
      {loading ? <div className="center-state">Loading workspaces…</div> : !workspaceId ? <div className="center-state"><h2>No workspace yet</h2><p>Open workspace settings in the sidebar to create one.</p></div> : !userId.trim() ? <div className="center-state">Enter a User ID in workspace settings.</div> : <>
        <div style={{ display: page === 'chat' ? 'flex' : 'none', minHeight: 0, flex: 1 }}><ChatPage workspaceId={workspaceId} userId={userId} /></div>
        {page === 'documents' && <DocumentsPage workspaceId={workspaceId} />}
      </>}
    </div>
  </div>
}
