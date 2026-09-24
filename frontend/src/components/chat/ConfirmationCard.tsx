import { AlertTriangle, Check, X } from 'lucide-react'
import type { Interrupt } from '../../types/api'

export function ConfirmationCard({ interrupt, busy, onDecision }: { interrupt: Interrupt; busy: boolean; onDecision: (decision: 'approve' | 'reject') => void }) {
  const action = interrupt.interrupt
  return <div className="confirmation-card" role="group" aria-label="操作需要确认">
    <div className="confirmation-icon"><AlertTriangle size={20} /></div>
    <div className="confirmation-content"><div className="confirmation-eyebrow">人工确认</div><h3>此操作需要你的确认</h3><p>{action.message || '请确认是否执行此操作。'}</p>
      <div className="action-details"><span>操作</span><strong>{action.action === 'delete_document' ? '删除文档' : action.action || '待确认操作'}</strong>{action.document_id != null && <><span>文档 ID</span><strong>#{action.document_id}</strong></>}{action.filename && <><span>文件</span><strong>{action.filename}</strong></>}</div>
      <div className="confirmation-actions"><button className="button-secondary" disabled={busy} onClick={() => onDecision('reject')}><X size={16} /> 拒绝</button><button className="button-danger" disabled={busy} onClick={() => onDecision('approve')}><Check size={16} /> 批准</button></div>
    </div>
  </div>
}
