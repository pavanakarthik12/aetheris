import type { ChatMessage } from "../../types/chat";
import { LoadingIndicator } from "./LoadingIndicator";

interface ChatWindowProps {
  messages: ChatMessage[];
  loading: boolean;
}

export function ChatWindow({ messages, loading }: ChatWindowProps) {
  return (
    <section className="chat-window" aria-live="polite" aria-label="Chat messages">
      {messages.length === 0 ? (
        <div className="chat-empty">
          <p style={{ margin: 0 }}>Send a message to start the conversation.</p>
        </div>
      ) : (
        <div className="chat-message-list">
          {messages.map((message) => (
            <article key={message.id} className={`chat-message chat-message--${message.role}`}>
              <span className="chat-message-label">{message.role}</span>
              <div className="chat-message-bubble">{message.content}</div>
            </article>
          ))}
          {loading ? <LoadingIndicator /> : null}
        </div>
      )}
    </section>
  );
}