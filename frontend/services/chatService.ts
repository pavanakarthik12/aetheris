import type { ChatRequest, ChatResponse } from "../types/chat";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export class ChatServiceError extends Error {
  statusCode: number;

  constructor(message: string, statusCode: number) {
    super(message);
    this.name = "ChatServiceError";
    this.statusCode = statusCode;
  }
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${BACKEND_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let message = "Unable to reach the backend.";
    const errorBody = await response.text();

    try {
      const payload = JSON.parse(errorBody) as { error?: { message?: string } };
      message = payload.error?.message ?? message;
    } catch {
      message = errorBody || message;
    }

    throw new ChatServiceError(message, response.status);
  }

  return (await response.json()) as ChatResponse;
}