interface ErrorNoticeProps {
  message: string | null;
}

export function ErrorNotice({ message }: ErrorNoticeProps) {
  if (!message) {
    return null;
  }

  return <div className="error-notice">{message}</div>;
}
