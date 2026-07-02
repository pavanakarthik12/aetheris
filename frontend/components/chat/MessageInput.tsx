interface MessageInputProps {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
}

export function MessageInput({ value, disabled, onChange }: MessageInputProps) {
  return (
    <input
      className="chat-input"
      type="text"
      placeholder="Type your message"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      disabled={disabled}
      aria-label="Message input"
    />
  );
}