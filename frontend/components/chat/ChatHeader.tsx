interface ChatHeaderProps {
  title: string;
  subtitle: string;
}

export function ChatHeader({ title, subtitle }: ChatHeaderProps) {
  return (
    <header className="chat-header">
      <h1 className="chat-title">{title}</h1>
      <p className="chat-subtitle">{subtitle}</p>
    </header>
  );
}