import { AlertTriangle, Check, X } from 'lucide-react'
import type { Interrupt } from '../../types/api'

export function ConfirmationCard({ interrupt, busy, onDecision }: { interrupt: Interrupt; busy: boolean; onDecision: (decision: 'approve' | 'reject') => void }) {
  const action = interrupt.interrupt
  return <div className="confirmation-card" role="group" aria-label="Action requires confirmation">
    <div className="confirmation-icon"><AlertTriangle size={20} /></div>
    <div className="confirmation-content"><div className="confirmation-eyebrow">HUMAN IN THE LOOP</div><h3>Action requires confirmation</h3><p>{action.message || 'This action needs your approval.'}</p>
      <div className="action-details"><span>Action</span><strong>{action.action === 'delete_document' ? 'Delete document' : action.action || 'Pending action'}</strong>{action.document_id != null && <><span>Document ID</span><strong>#{action.document_id}</strong></>}{action.filename && <><span>File</span><strong>{action.filename}</strong></>}</div>
      <div className="confirmation-actions"><button className="button-secondary" disabled={busy} onClick={() => onDecision('reject')}><X size={16} /> Reject</button><button className="button-danger" disabled={busy} onClick={() => onDecision('approve')}><Check size={16} /> Approve</button></div>
    </div>
  </div>
}
