import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowUp, MessageSquare, Plus, Sparkles } from 'lucide-react'
import { ConfirmationCard } from '../components/chat/ConfirmationCard'
import { MessageBubble } from '../components/chat/MessageBubble'
import { createThread, listMessages, listThreads, streamChat } from '../services/chat'
import type { ChatMessage, Interrupt, Source, Thread } from '../types/api'

const storageKey = (workspaceId: number, userId: string) => `nodagent.thread.${workspaceId}.${userId}`
const interruptKey = (workspaceId: number, userId: string, threadId: string) => `nodagent.interrupt.${workspaceId}.${userId}.${threadId}`
const suggestions = ['Python 的装饰器是什么？', '总结知识库中的文档', '东京现在天气怎么样？', '查看 21breeze/NodAgent 仓库信息']

export function ChatPage({ workspaceId, userId }: { workspaceId: number; userId: string }) {
  const [threads, setThreads] = useState<Thread[]>([])
  const [thread, setThread] = useState<Thread | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const newThreadRef = useRef<string | null>(null)

  useEffect(() => {
    let active = true
    abortRef.current?.abort()
    setThread(null); setMessages([]); setError(''); setBusy(false)
    listThreads(workspaceId, userId).then(items => {
      if (!active) return
      setThreads(items)
      const saved = localStorage.getItem(storageKey(workspaceId, userId))
      setThread(items.find(item => item.thread_id === saved) || items[0] || null)
    }).catch(err => { if (active) setError(err instanceof Error ? err.message : String(err)) })
    return () => { active = false; abortRef.current?.abort() }
  }, [workspaceId, userId])

  useEffect(() => {
    if (!thread) { setMessages([]); return }
    localStorage.setItem(storageKey(workspaceId, userId), thread.thread_id)
    if (newThreadRef.current === thread.thread_id) { newThreadRef.current = null; return }
    let active = true
    listMessages(workspaceId, thread.thread_id, userId).then(items => {
      if (!active) return
      const restored = localStorage.getItem(interruptKey(workspaceId, userId, thread.thread_id))
      let interrupt: Interrupt | null = null
      try { if (restored) interrupt = JSON.parse(restored) as Interrupt } catch { localStorage.removeItem(interruptKey(workspaceId, userId, thread.thread_id)) }
      setMessages([...items.map(item => ({ key: String(item.id), role: item.role === 'user' ? 'user' as const : 'assistant' as const, content: item.content, sources: item.sources || [] })), ...(interrupt ? [{ key: interrupt.interrupt_id, role: 'assistant' as const, content: '', sources: [], interrupt }] : [])])
    }).catch(err => { if (active) setError(err instanceof Error ? err.message : String(err)) })
    return () => { active = false }
  }, [workspaceId, userId, thread?.thread_id])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const newThread = useCallback(async () => {
    if (busy) return null
    try {
      const created = await createThread(workspaceId, userId)
      newThreadRef.current = created.thread_id
      setThreads(items => [created, ...items])
      setThread(created); setMessages([]); setError('')
      return created
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); return null }
  }, [workspaceId, userId, busy])

  async function runStream(activeThread: Thread, key: string, payload: { message?: string; decision?: 'approve' | 'reject' }) {
    const controller = new AbortController()
    abortRef.current = controller
    setBusy(true); setError('')
    let terminal = false
    try {
      await streamChat(workspaceId, { user_id: userId, thread_id: activeThread.thread_id, ...payload }, ({ event, data }) => {
        const value = data as Record<string, unknown>
        if (event === 'token') setMessages(items => items.map(item => item.key === key ? { ...item, content: item.content + String(value.content || '') } : item))
        if (event === 'sources') setMessages(items => items.map(item => item.key === key ? { ...item, sources: (value.sources as Source[]) || [] } : item))
        if (event === 'interrupt') { terminal = true; localStorage.setItem(interruptKey(workspaceId, userId, activeThread.thread_id), JSON.stringify(value)); setMessages(items => items.map(item => item.key === key ? { ...item, pending: false, interrupt: value as unknown as Interrupt } : item)) }
        if (event === 'resume') setMessages(items => items.map(item => item.key === key ? { ...item, interrupt: undefined, content: '', sources: [] } : item))
        if (event === 'done') { terminal = true; localStorage.removeItem(interruptKey(workspaceId, userId, activeThread.thread_id)); setMessages(items => items.map(item => item.key === key ? { ...item, pending: false, interrupt: undefined } : item)); void listThreads(workspaceId, userId).then(setThreads) }
        if (event === 'error') { terminal = true; setMessages(items => items.map(item => item.key === key ? { ...item, pending: false, error: String(value.message || 'Agent failed') } : item)) }
      }, controller.signal)
    } catch (err) {
      if (!controller.signal.aborted) setMessages(items => items.map(item => item.key === key ? { ...item, pending: false, error: err instanceof Error ? err.message : String(err) } : item))
    } finally {
      if (!terminal && controller.signal.aborted) setMessages(items => items.map(item => item.key === key ? { ...item, pending: false } : item))
      setBusy(false); abortRef.current = null
    }
  }

  async function send(input = draft) {
    const message = input.trim()
    if (!message || busy) return
    let activeThread = thread
    if (!activeThread) activeThread = await newThread()
    if (!activeThread) return
    const userKey = crypto.randomUUID(), aiKey = crypto.randomUUID()
    setMessages(items => [...items, { key: userKey, role: 'user', content: message, sources: [] }, { key: aiKey, role: 'assistant', content: '', sources: [], pending: true }])
    setDraft('')
    await runStream(activeThread, aiKey, { message })
  }

  async function decide(key: string, decision: 'approve' | 'reject') {
    if (!thread || busy) return
    setMessages(items => items.map(item => item.key === key ? { ...item, pending: true, error: undefined } : item))
    await runStream(thread, key, { decision })
  }

  const pending = messages.find(message => message.interrupt)
  return <div className="chat-layout">
    <div className="chat-heading"><div><div className="eyebrow">AI ASSISTANT</div><h1>Chat</h1><p>Ask anything, search your knowledge base, or work with connected tools.</p></div><button className="button-outline" onClick={() => void newThread()} disabled={busy}><Plus size={16} /> New thread</button></div>
    <div className="thread-strip"><MessageSquare size={14} /><span>THREAD</span><select aria-label="Current thread" value={thread?.thread_id || ''} onChange={event => setThread(threads.find(item => item.thread_id === event.target.value) || null)} disabled={busy}><option value="">New conversation</option>{threads.map(item => <option value={item.thread_id} key={item.thread_id}>{item.title} · {item.thread_id.slice(0, 8)}</option>)}</select>{thread && <code title={thread.thread_id}>{thread.thread_id.slice(0, 8)}</code>}</div>
    <div className="chat-scroll">
      {messages.length === 0 ? <div className="chat-empty"><div className="empty-symbol"><Sparkles size={27} /></div><h2>What can I help you explore?</h2><p>Connect ideas across your documents, ask a general question, or use an external tool. Your workspace is ready.</p><div className="suggestions">{suggestions.map(suggestion => <button key={suggestion} onClick={() => void send(suggestion)}>{suggestion}<span>↗</span></button>)}</div></div> : <div className="message-list">{messages.map(message => <div key={message.key}><MessageBubble message={message} />{message.interrupt && <ConfirmationCard interrupt={message.interrupt} busy={busy} onDecision={decision => void decide(message.key, decision)} />}</div>)}<div ref={bottomRef} /></div>}
    </div>
    <div className="composer-wrap">{error && <div className="inline-error" role="alert">{error}</div>}{pending && <div className="composer-note">Respond to the confirmation above to continue this thread.</div>}<div className="composer"><textarea aria-label="Message" placeholder="Ask NodAgent anything..." rows={2} value={draft} disabled={busy || !!pending} onChange={event => setDraft(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void send() } }} /><button className="send-button" aria-label="Send message" disabled={!draft.trim() || busy || !!pending} onClick={() => void send()}><ArrowUp size={19} /></button></div><div className="composer-footer"><span>Enter to send · Shift + Enter for a new line</span><span>Powered by LangGraph</span></div></div>
  </div>
}
