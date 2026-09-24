import { apiUrl, jsonOptions, request, responseError } from './api'
import type { HistoryMessage, Thread } from '../types/api'

const base = (workspaceId: number) => `/workspaces/${workspaceId}`
export const listThreads = (workspaceId: number, userId: string) => request<Thread[]>(`${base(workspaceId)}/threads?user_id=${encodeURIComponent(userId)}`)
export const createThread = (workspaceId: number, userId: string) => request<Thread>(`${base(workspaceId)}/threads`, jsonOptions({ user_id: userId, thread_id: crypto.randomUUID() }))
export const listMessages = (workspaceId: number, threadId: string, userId: string) => request<HistoryMessage[]>(`${base(workspaceId)}/threads/${encodeURIComponent(threadId)}/messages?user_id=${encodeURIComponent(userId)}`)

export type StreamEvent = { event: string; data: unknown }

// POST streaming cannot use EventSource; decode the response body as SSE frames.
export async function streamChat(
  workspaceId: number,
  payload: { user_id: string; thread_id: string; message?: string; decision?: 'approve' | 'reject' },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resume = payload.decision !== undefined
  const endpoint = `${base(workspaceId)}/chat/stream${resume ? '/resume' : ''}`
  const response = await fetch(apiUrl(endpoint), { ...jsonOptions(payload), signal, headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' } })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error('浏览器未收到流式响应')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let terminal = false
  const dispatch = (frame: string) => {
    let event = 'message'
    const data: string[] = []
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
    }
    if (!data.length) return
    const raw = data.join('\n')
    let parsed: unknown
    try { parsed = JSON.parse(raw) } catch { parsed = raw }
    onEvent({ event, data: parsed })
    if (event === 'done' || event === 'interrupt' || event === 'error') terminal = true
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let boundary: RegExpExecArray | null
    while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
      dispatch(buffer.slice(0, boundary.index).replace(/\r\n/g, '\n'))
      buffer = buffer.slice(boundary.index + boundary[0].length)
    }
  }
  buffer += decoder.decode()
  if (buffer.trim()) dispatch(buffer.replace(/\r\n/g, '\n'))
  if (!terminal) throw new Error('连接中断，未收到完成事件。请查看会话记录后再重试。')
}
