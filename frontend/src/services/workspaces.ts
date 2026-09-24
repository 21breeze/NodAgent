import { jsonOptions, request } from './api'
import type { Workspace } from '../types/api'

export const listWorkspaces = () => request<Workspace[]>('/workspaces')
export const createWorkspace = (name: string) => request<Workspace>('/workspaces', jsonOptions({ name }))
