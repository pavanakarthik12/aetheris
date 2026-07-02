interface SendButtonProps {
  loading: boolean;
  disabled: boolean;
}

export function SendButton({ loading, disabled }: SendButtonProps) {
  return (
    <button className="chat-button" type="submit" disabled={disabled}>
      {loading ? "Sending..." : "Send"}
    </button>
  );
}