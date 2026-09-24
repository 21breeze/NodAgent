const baseUrl = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

export function apiUrl(path: string): string { return `${baseUrl}/api${path}` }

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), options)
  if (!response.ok) throw new Error(await responseError(response))
  return response.json() as Promise<T>
}

export async function responseError(response: Response): Promise<string> {
  try {
    const body = await response.json() as { detail?: string | { msg?: string }[] }
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) return body.detail.map(item => item.msg).join('; ')
  } catch { /* response is not JSON */ }
  return `请求失败 (${response.status})`
}

export function jsonOptions(body: unknown): RequestInit {
  return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}
