import { useCallback, useEffect, useState } from 'react'
import { BookOpen, ChevronDown, MessageSquare, Plus, Settings2, Sparkles } from 'lucide-react'
import { ChatPage } from './pages/ChatPage'
import { DocumentsPage } from './pages/DocumentsPage'
import { createWorkspace, listWorkspaces } from './services/workspaces'
import type { Thread, Workspace } from './types/api'

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
  const [recentThreads, setRecentThreads] = useState<Thread[]>([])
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null)
  const [newChatRequest, setNewChatRequest] = useState(0)

  useEffect(() => {
    listWorkspaces().then(items => {
      setWorkspaces(items)
      setWorkspaceId(current => items.find(item => item.id === current)?.id ?? items[0]?.id ?? null)
      setError('')
    }).catch(err => setError(err instanceof Error ? err.message : String(err))).finally(() => setLoading(false))
  }, [])
  useEffect(() => { if (workspaceId) localStorage.setItem('nodagent.workspace', String(workspaceId)) }, [workspaceId])
  useEffect(() => { localStorage.setItem('nodagent.user', userId) }, [userId])
  useEffect(() => { setRecentThreads([]); setActiveThreadId(null) }, [workspaceId, userId])

  const workspace = workspaces.find(item => item.id === workspaceId)
  const updateThreads = useCallback((items: Thread[]) => setRecentThreads(items), [])
  const updateThread = useCallback((id: string | null) => setActiveThreadId(id), [])

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

  function openNewChat() {
    setPage('chat')
    setNewChatRequest(value => value + 1)
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Sparkles size={21} strokeWidth={2} /></span><span>NodAgent</span></div>
      <button className="sidebar-new-chat" onClick={openNewChat}><Plus size={19} /><span>发起新对话</span></button>
      <nav className="nav-list" aria-label="主导航">
        <button className={page === 'chat' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('chat')}><MessageSquare size={17} /><span>对话</span></button>
        <button className={page === 'documents' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('documents')}><BookOpen size={17} /><span>知识库文档</span></button>
      </nav>
      <div className="sidebar-section-label">最近对话</div>
      <div className="recent-list">
        {recentThreads.length === 0 ? <p className="recent-empty">暂无对话记录</p> : recentThreads.slice(0, 12).map(item => <button key={item.thread_id} title={item.title} className={`recent-item ${activeThreadId === item.thread_id && page === 'chat' ? 'selected' : ''}`} onClick={() => { setPage('chat'); setActiveThreadId(item.thread_id) }}><MessageSquare size={15} /><span>{item.title || '新对话'}</span></button>)}
      </div>
      <div className="sidebar-bottom">
        <div className="workspace-caption">当前工作区</div>
        <button className="workspace-switch" onClick={() => setSettingsOpen(value => !value)} aria-expanded={settingsOpen}>
          <span className="workspace-avatar">{workspace?.name?.slice(0, 1).toUpperCase() || 'N'}</span>
          <span className="workspace-copy"><strong>{workspace?.name || (loading ? '加载中…' : '选择工作区')}</strong><small>{workspace ? `工作区 #${workspace.id}` : '开始使用'}</small></span>
          <ChevronDown size={15} />
        </button>
        {settingsOpen && <div className="workspace-panel">
          <label htmlFor="workspace-select">切换工作区</label>
          <select id="workspace-select" value={workspaceId ?? ''} onChange={event => setWorkspaceId(Number(event.target.value))}>
            {workspaces.map(item => <option key={item.id} value={item.id}>{item.name} (#{item.id})</option>)}
          </select>
          <label htmlFor="new-workspace">新建工作区</label>
          <div className="inline-form"><input id="new-workspace" placeholder="工作区名称" value={newWorkspaceName} onChange={event => setNewWorkspaceName(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') void addWorkspace() }} /><button onClick={() => void addWorkspace()} aria-label="创建工作区"><Plus size={16} /></button></div>
          <label htmlFor="user-id">用户 ID</label>
          <input id="user-id" value={userId} onChange={event => setUserId(event.target.value)} maxLength={100} />
        </div>}
        <button className="sidebar-settings" onClick={() => setSettingsOpen(value => !value)}><Settings2 size={16} /><span>工作区设置</span></button>
      </div>
    </aside>
    <div className="main-area">
      <header className="topbar"><div className="topbar-title">NodAgent <ChevronDown size={15} /></div><div className="topbar-right"><span className="workspace-pill">{workspace?.name || '未选择工作区'}</span><span className="user-pill" title={`用户：${userId}`}>{userId.slice(0, 1).toUpperCase() || 'U'}</span></div></header>
      {error && <div className="global-error" role="alert">{error}<button onClick={() => setError('')}>关闭</button></div>}
      {loading ? <div className="center-state">正在加载工作区…</div> : !workspaceId ? <div className="center-state"><h2>还没有工作区</h2><p>打开左下角的工作区设置，创建一个工作区即可开始。</p></div> : !userId.trim() ? <div className="center-state">请在工作区设置中填写用户 ID。</div> : <>
        <div style={{ display: page === 'chat' ? 'flex' : 'none', minHeight: 0, flex: 1 }}><ChatPage workspaceId={workspaceId} userId={userId} selectedThreadId={activeThreadId} newChatRequest={newChatRequest} onThreadsChange={updateThreads} onThreadChange={updateThread} /></div>
        {page === 'documents' && <DocumentsPage workspaceId={workspaceId} />}
      </>}
    </div>
  </div>
}
