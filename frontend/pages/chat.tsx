import { FormEvent, useEffect, useRef, useState } from "react";

import { ChatHeader } from "../components/chat/ChatHeader";
import { ChatWindow } from "../components/chat/ChatWindow";
import { MessageInput } from "../components/chat/MessageInput";
import { SendButton } from "../components/chat/SendButton";
import { sendChatMessage, ChatServiceError } from "../services/chatService";
import type { ChatMessage } from "../types/chat";

function createMessage(role: ChatMessage["role"], content: string): ChatMessage {
  return {
    id: globalThis.crypto.randomUUID(),
    role,
    content,
  };
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endOfConversationRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endOfConversationRef.current?.scrollIntoView({ block: "end" });
  }, [messages, loading]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedMessage = message.trim();
    if (!trimmedMessage || loading) {
      return;
    }

    setError(null);
    setLoading(true);
    setMessage("");

    const userMessage = createMessage("user", trimmedMessage);
    setMessages((currentMessages) => [...currentMessages, userMessage]);

    if (process.env.NODE_ENV !== "production") {
      console.log("Aetheris chat request", trimmedMessage);
    }

    try {
      const response = await sendChatMessage({ message: trimmedMessage });
      const assistantMessage = createMessage("assistant", response.response);
      setMessages((currentMessages) => [...currentMessages, assistantMessage]);

      if (process.env.NODE_ENV !== "production") {
        console.log("Aetheris chat response", response.response);
      }
    } catch (caughtError) {
      const errorMessage =
        caughtError instanceof ChatServiceError
          ? caughtError.message
          : "Something went wrong while contacting the backend.";

      setError(errorMessage);

      if (process.env.NODE_ENV !== "production") {
        console.error("Aetheris chat error", caughtError);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <div className="chat-shell">
        <section className="chat-card">
          <ChatHeader
            title="Aetheris Chat"
            subtitle="A minimal Phase 1 interface for sending one message to Qwen and displaying the response."
          />

          <ChatWindow messages={messages} loading={loading} />

          {error ? <p className="chat-error">{error}</p> : null}

          <footer className="chat-footer">
            <form className="chat-form" onSubmit={handleSubmit}>
              <MessageInput value={message} disabled={loading} onChange={setMessage} />
              <SendButton loading={loading} disabled={loading || !message.trim()} />
            </form>
            <p className="muted" style={{ margin: 0 }}>
              Press Enter to send.
            </p>
            <div ref={endOfConversationRef} />
          </footer>
        </section>
      </div>
    </main>
  );
}