interface MessageInputProps {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onKeyDown?: (event: React.KeyboardEvent<HTMLInputElement>) => void;
}

export function MessageInput({ value, disabled, onChange, onKeyDown }: MessageInputProps) {
  return (
    <input
      className="chat-input"
      type="text"
      placeholder="Type your message"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      onKeyDown={onKeyDown}
      disabled={disabled}
      aria-label="Message input"
    />
  );
}
