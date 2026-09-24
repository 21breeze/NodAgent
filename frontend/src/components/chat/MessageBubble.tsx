import { Bot, ChevronDown, FileText, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage, Source } from '../../types/api'

function SourceCard({ source }: { source: Source }) {
  return <div className="source-card">
    <div className="source-title"><span className="source-number">{source.citation}</span><FileText size={15} /><strong>{source.filename}</strong></div>
    <div className="source-meta"><span>Chunk {source.chunk_index + 1}</span>{source.page_number != null && <span>Page {source.page_number}</span>}{source.similarity != null && <span>Similarity {source.similarity.toFixed(2)}</span>}</div>
    <p>{source.content}</p>
  </div>
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const assistant = message.role === 'assistant'
  return <article className={`message-row ${assistant ? 'assistant-row' : 'user-row'}`}>
    <div className={`message-avatar ${assistant ? 'ai-avatar' : 'human-avatar'}`}>{assistant ? <Bot size={17} /> : <User size={17} />}</div>
    <div className="message-body"><div className="message-author">{assistant ? 'NodAgent' : 'You'}</div>
      {message.content ? <div className="message-content markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div> : message.pending && !message.interrupt ? <div className="typing"><span /><span /><span /></div> : null}
      {message.error && <div className="message-error" role="alert">{message.error}</div>}
      {message.pending && message.content && !message.interrupt && <span className="stream-cursor" aria-label="Generating" />}
      {message.sources.length > 0 && <details className="sources"><summary><FileText size={15} /> Sources ({message.sources.length}) <ChevronDown size={15} className="source-chevron" /></summary><div className="source-list">{message.sources.map(source => <SourceCard key={source.chunk_id} source={source} />)}</div></details>}
    </div>
  </article>
}
