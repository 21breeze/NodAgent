import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowRight, BookOpen, FileText, LoaderCircle, RefreshCw, RotateCw, Trash2, UploadCloud } from 'lucide-react'
import { getDocumentChunks, getDocumentStatus, listDocuments, processDocument, uploadDocument } from '../services/documents'
import type { Document, DocumentChunk, DocumentStatus } from '../types/api'

const activeStatuses: DocumentStatus[] = ['uploaded', 'queued', 'processing']
const statusLabel: Record<DocumentStatus, string> = { uploaded: '已上传', queued: '排队中', processing: '处理中', completed: '已完成', failed: '失败' }

export function DocumentsPage({ workspaceId, onDeleteRequest }: { workspaceId: number; onDeleteRequest: (documentId: number) => void }) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [uploadNotice, setUploadNotice] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const [chunks, setChunks] = useState<DocumentChunk[] | null>(null)
  const [chunkError, setChunkError] = useState('')
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try { setDocuments(await listDocuments(workspaceId)) }
    catch (err) { setError(err instanceof Error ? err.message : String(err)) }
    finally { setLoading(false) }
  }, [workspaceId])

  useEffect(() => { setLoading(true); setSelected(null); setChunks(null); void refresh() }, [refresh])

  // One interval per workspace; only active documents are polled, and it ends when none remain.
  const activeIds = documents.filter(doc => activeStatuses.includes(doc.processing_status)).map(doc => doc.id).join(',')
  useEffect(() => {
    if (!activeIds) return
    let cancelled = false
    let running = false
    const ids = activeIds.split(',').map(Number)
    const tick = async () => {
      if (running) return
      running = true
      try {
        const results = await Promise.allSettled(ids.map(id => getDocumentStatus(workspaceId, id)))
        if (cancelled) return
        let finished = false
        setDocuments(items => items.map(doc => {
          const index = ids.indexOf(doc.id)
          const result = results[index]
          if (!result || result.status !== 'fulfilled') return doc
          const status = result.value
          if (status.status === 'completed' || status.status === 'failed') finished = true
          return { ...doc, processing_status: status.status, processing_task_id: status.task_id, processing_error: status.error }
        }))
        if (finished) void refresh()
        const rejected = results.find(result => result.status === 'rejected')
        if (rejected?.status === 'rejected') setError(rejected.reason instanceof Error ? rejected.reason.message : String(rejected.reason))
      } finally { running = false }
    }
    void tick()
    const timer = window.setInterval(() => void tick(), 2000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [workspaceId, activeIds, refresh])

  async function upload(file: File) {
    if (!/\.(pdf|txt|md)$/i.test(file.name)) { setError('仅支持 PDF、TXT、MD 文件。'); return }
    setUploading(true); setError(''); setUploadNotice('')
    try { const created = await uploadDocument(workspaceId, file); setDocuments(items => [created, ...items.filter(item => item.id !== created.id)]); setUploadNotice(`「${file.name}」已上传，正在处理。`); void refresh() }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); void refresh() }
    finally { setUploading(false); if (fileInput.current) fileInput.current.value = '' }
  }

  async function openChunks(id: number) {
    if (selected === id) { setSelected(null); setChunks(null); return }
    setSelected(id); setChunks(null); setChunkError('')
    try { setChunks(await getDocumentChunks(workspaceId, id)) }
    catch (err) { setChunkError(err instanceof Error ? err.message : String(err)) }
  }

  async function retry(id: number) {
    try {
      const status = await processDocument(workspaceId, id)
      setDocuments(items => items.map(doc => doc.id === id ? { ...doc, processing_status: status.status, processing_task_id: status.task_id, processing_error: status.error } : doc))
      setError('')
    } catch (err) { setError(err instanceof Error ? err.message : String(err)) }
  }

  const completed = documents.filter(doc => doc.processing_status === 'completed').length
  const processing = documents.filter(doc => activeStatuses.includes(doc.processing_status)).length
  return <div className="documents-page">
    <div className="page-heading"><div><h1>知识库文档</h1><p>上传资料，让 NodAgent 基于你的知识库回答问题。</p></div><button className="button-outline" onClick={() => void refresh()}><RefreshCw size={16} /> 刷新</button></div>
    <div className="document-stats"><div><span>全部文档</span><strong>{documents.length}</strong></div><div><span>可检索</span><strong>{completed}</strong></div><div><span>处理中</span><strong>{processing}</strong></div></div>
    <input ref={fileInput} className="sr-only" type="file" accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown" onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file) }} />
    <div className={`upload-zone ${dragging ? 'dragging' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files[0]; if (file) void upload(file) }}>
      <div className="upload-icon">{uploading ? <LoaderCircle size={23} className="spin" /> : <UploadCloud size={23} />}</div><div><strong>{uploading ? '正在上传文档…' : '添加知识库文档'}</strong><p>拖拽文件到这里，或点击选择文件。支持 PDF、TXT、MD，单文件不超过 100 MB。</p></div><button disabled={uploading} className="button-primary" onClick={() => fileInput.current?.click()}>选择文件 <ArrowRight size={16} /></button>
    </div>
    {uploadNotice && <div className="inline-success" role="status">{uploadNotice}</div>}
    {error && <div className="inline-error" role="alert">{error}</div>}
    <div className="list-heading"><div><h2>全部文档</h2><p>处理状态每 2 秒自动更新</p></div><span>共 {documents.length} 个文件</span></div>
    <div className="documents-list">{loading ? <div className="list-empty">正在加载文档…</div> : documents.length === 0 ? <div className="list-empty"><BookOpen size={24} /><p>还没有文档。上传文件，开始构建你的知识库。</p></div> : documents.map(doc => <div className="document-item" key={doc.id}>
      <div className="document-main"><span className="document-icon"><FileText size={21} /></span><div className="document-name"><strong>{doc.filename}</strong><span>文档 #{doc.id} · {doc.filename.split('.').pop()?.toUpperCase()}</span></div><span className={`status-badge status-${doc.processing_status}`}><span />{statusLabel[doc.processing_status] || doc.processing_status}</span><button className="icon-button" aria-label={`查看 ${doc.filename} 的片段`} title="查看片段" onClick={() => void openChunks(doc.id)}><ArrowRight size={17} /></button><button className="icon-button delete-icon" aria-label={`删除 ${doc.filename}`} title={activeStatuses.includes(doc.processing_status) ? '文档处理中，暂时不能删除' : '发起删除（需要人工确认）'} disabled={activeStatuses.includes(doc.processing_status)} onClick={() => onDeleteRequest(doc.id)}><Trash2 size={17} /></button></div>
      {doc.processing_error && <div className="document-error">{doc.processing_error}</div>}
      {doc.processing_status === 'failed' && <button className="retry-button" onClick={() => void retry(doc.id)}><RotateCw size={14} /> 重新处理</button>}
      {selected === doc.id && <div className="chunks-panel"><h3>文档片段</h3>{chunkError ? <div className="inline-error">{chunkError}</div> : chunks === null ? <p>正在加载片段…</p> : chunks.length === 0 ? <p>暂无片段。</p> : chunks.map(chunk => <div className="chunk" key={chunk.id}><span>片段 {chunk.chunk_index + 1}{typeof chunk.metadata_json.page_number === 'number' ? ` · 第 ${chunk.metadata_json.page_number} 页` : ''}</span><p>{chunk.content}</p></div>)}</div>}
    </div>)}</div>
  </div>
}
