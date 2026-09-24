export interface Workspace { id: number; name: string; description: string | null }
export interface Thread { id: number; workspace_id: number; user_id: string; thread_id: string; title: string; created_at: string; updated_at: string }
export interface Source {
  source_number: number; citation: string; chunk_id: number; document_id: number;
  filename: string; chunk_index: number; page_number: number | null; content: string;
  metadata_json: Record<string, unknown>; distance: number | null; similarity: number | null;
  keyword_score: number | null; vector_rank: number | null; keyword_rank: number | null; rrf_score: number | null;
}
export interface HistoryMessage { id: number; chat_thread_id: number; role: string; content: string; sources: Source[]; created_at: string }
export interface Interrupt { status: 'interrupted'; thread_id: string; interrupt_id: string; interrupt: { type?: string; action?: string; document_id?: number; filename?: string; message?: string } }
export interface Document { id: number; workspace_id: number; filename: string; file_path: string; content_type: string | null; processing_status: DocumentStatus; processing_task_id: string | null; processing_error: string | null }
export type DocumentStatus = 'uploaded' | 'queued' | 'processing' | 'completed' | 'failed'
export interface DocumentProcessing { document_id: number; status: DocumentStatus; task_id: string | null; error: string | null }
export interface DocumentChunk { id: number; workspace_id: number; document_id: number; chunk_index: number; content: string; metadata_json: Record<string, unknown> }
export interface ChatMessage { key: string; role: 'user' | 'assistant'; content: string; sources: Source[]; pending?: boolean; interrupt?: Interrupt; error?: string }
