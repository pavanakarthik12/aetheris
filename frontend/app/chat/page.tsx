"use client";

import { useCallback, useState } from "react";

import { ChatHeader } from "../../components/chat/ChatHeader";
import { ChatWindow } from "../../components/chat/ChatWindow";
import { MessageInput } from "../../components/chat/MessageInput";
import { SendButton } from "../../components/chat/SendButton";
import type { ChatMessage } from "../../types/chat";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const appendMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const handleSubmit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();

      const text = input.trim();
      if (!text || loading) return;

      setError(null);
      setInput("");

      const userMessage: ChatMessage = {
        id: generateId(),
        role: "user",
        content: text,
      };
      appendMessage(userMessage);
      setLoading(true);

      try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          const detail =
            payload?.error?.message ??
            payload?.detail ??
            `Server error (${response.status})`;
          throw new Error(detail);
        }

        const data = await response.json();
        const assistantMessage: ChatMessage = {
          id: generateId(),
          role: "assistant",
          content: data.response,
        };
        appendMessage(assistantMessage);
      } catch (err) {
        const message = err instanceof Error ? err.message : "An unexpected error occurred.";
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, appendMessage],
  );

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === "Enter" && !event.shiftKey) {
        // Let the browser bubble to the form's submit handler naturally
        // by doing nothing — the form has type="submit" on the button.
        // We only prevent default here to stop a bare newline being typed.
        event.preventDefault();
        const form = (event.currentTarget as HTMLElement).closest("form");
        form?.requestSubmit();
      }
    },
    [],
  );

  const isDisabled = loading || input.trim().length === 0;

  return (
    <main>
      <div className="chat-shell">
        <div className="chat-card">
          <ChatHeader
            title="Aetheris"
            subtitle="Cognitive AI — Phase 1 Chat Interface"
          />

          <ChatWindow messages={messages} loading={loading} />

          {error ? (
            <p className="chat-error" role="alert">
              {error}
            </p>
          ) : null}

          <footer className="chat-footer">
            <form className="chat-form" onSubmit={handleSubmit} noValidate>
              <MessageInput
                value={input}
                disabled={loading}
                onChange={setInput}
                onKeyDown={handleKeyDown}
              />
              <SendButton loading={loading} disabled={isDisabled} />
            </form>
          </footer>
        </div>
      </div>
    </main>
  );
}
