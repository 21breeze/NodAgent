import { request } from './api'
import type { Document, DocumentChunk, DocumentProcessing } from '../types/api'

const path = (workspaceId: number) => `/workspaces/${workspaceId}/documents`
export const listDocuments = (workspaceId: number) => request<Document[]>(path(workspaceId))
export const getDocumentStatus = (workspaceId: number, id: number) => request<DocumentProcessing>(`${path(workspaceId)}/${id}/status`)
export const getDocumentChunks = (workspaceId: number, id: number) => request<DocumentChunk[]>(`${path(workspaceId)}/${id}/chunks`)
export const processDocument = (workspaceId: number, id: number) => request<DocumentProcessing>(`${path(workspaceId)}/${id}/process`, { method: 'POST' })
export function uploadDocument(workspaceId: number, file: File) {
  const body = new FormData()
  body.append('file', file)
  return request<Document>(path(workspaceId), { method: 'POST', body })
}
