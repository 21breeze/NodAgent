import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowRight, BookOpen, FileText, LoaderCircle, RefreshCw, RotateCw, UploadCloud } from 'lucide-react'
import { getDocumentChunks, getDocumentStatus, listDocuments, processDocument, uploadDocument } from '../services/documents'
import type { Document, DocumentChunk, DocumentStatus } from '../types/api'

const activeStatuses: DocumentStatus[] = ['uploaded', 'queued', 'processing']
const statusLabel: Record<DocumentStatus, string> = { uploaded: 'Uploaded', queued: 'Queued', processing: 'Processing', completed: 'Completed', failed: 'Failed' }

export function DocumentsPage({ workspaceId }: { workspaceId: number }) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const [chunks, setChunks] = useState<DocumentChunk[] | null>(null)
  const [chunkError, setChunkError] = useState('')
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try { setDocuments(await listDocuments(workspaceId)); setError('') }
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
    if (!/\.(pdf|txt|md)$/i.test(file.name)) { setError('Supported files: PDF, TXT, MD.'); return }
    setUploading(true); setError('')
    try { const created = await uploadDocument(workspaceId, file); setDocuments(items => [created, ...items.filter(item => item.id !== created.id)]); void refresh() }
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
    <div className="page-heading"><div><div className="eyebrow">KNOWLEDGE BASE</div><h1>Documents</h1><p>Upload and organize the knowledge behind your agent.</p></div><button className="button-outline" onClick={() => void refresh()}><RefreshCw size={16} /> Refresh</button></div>
    <div className="document-stats"><div><span>TOTAL DOCUMENTS</span><strong>{documents.length}</strong></div><div><span>READY TO SEARCH</span><strong>{completed}</strong></div><div><span>IN PROGRESS</span><strong>{processing}</strong></div></div>
    <input ref={fileInput} className="sr-only" type="file" accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown" onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file) }} />
    <div className={`upload-zone ${dragging ? 'dragging' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files[0]; if (file) void upload(file) }}>
      <div className="upload-icon">{uploading ? <LoaderCircle size={23} className="spin" /> : <UploadCloud size={23} />}</div><div><strong>{uploading ? 'Uploading document…' : 'Add knowledge to your workspace'}</strong><p>Drag and drop a file here, or browse your computer. PDF, TXT and MD supported.</p></div><button disabled={uploading} className="button-primary" onClick={() => fileInput.current?.click()}>Browse files <ArrowRight size={16} /></button>
    </div>
    {error && <div className="inline-error" role="alert">{error}</div>}
    <div className="list-heading"><div><h2>All documents</h2><p>Processing updates automatically every 2 seconds.</p></div><span>{documents.length} files</span></div>
    <div className="documents-list">{loading ? <div className="list-empty">Loading documents…</div> : documents.length === 0 ? <div className="list-empty"><BookOpen size={24} /><p>No documents yet. Upload a file to start building your knowledge base.</p></div> : documents.map(doc => <div className="document-item" key={doc.id}>
      <div className="document-main"><span className="document-icon"><FileText size={21} /></span><div className="document-name"><strong>{doc.filename}</strong><span>Document #{doc.id} · {doc.filename.split('.').pop()?.toUpperCase()}</span></div><span className={`status-badge status-${doc.processing_status}`}><span />{statusLabel[doc.processing_status] || doc.processing_status}</span><button className="icon-button" aria-label={`View chunks for ${doc.filename}`} title="View chunks" onClick={() => void openChunks(doc.id)}><ArrowRight size={17} /></button></div>
      {doc.processing_error && <div className="document-error">{doc.processing_error}</div>}
      {doc.processing_status === 'failed' && <button className="retry-button" onClick={() => void retry(doc.id)}><RotateCw size={14} /> Retry processing</button>}
      {selected === doc.id && <div className="chunks-panel"><h3>Document chunks</h3>{chunkError ? <div className="inline-error">{chunkError}</div> : chunks === null ? <p>Loading chunks…</p> : chunks.length === 0 ? <p>No chunks available yet.</p> : chunks.map(chunk => <div className="chunk" key={chunk.id}><span>CHUNK {chunk.chunk_index + 1}{typeof chunk.metadata_json.page_number === 'number' ? ` · PAGE ${chunk.metadata_json.page_number}` : ''}</span><p>{chunk.content}</p></div>)}</div>}
    </div>)}</div>
  </div>
}
