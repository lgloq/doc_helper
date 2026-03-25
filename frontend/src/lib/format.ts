export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export function formatBytes(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function truncate(value: string, maxLength = 140): string {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 1)}...`;
}

export function locationLabel(input: {
  page_number_start?: number | null;
  paragraph_start?: number | null;
  chunk_index?: number | null;
}): string {
  if (input.page_number_start != null) {
    return `p.${input.page_number_start}`;
  }
  if (input.paragraph_start != null) {
    return `para.${input.paragraph_start}`;
  }
  if (input.chunk_index != null) {
    return `chunk ${input.chunk_index}`;
  }
  return "source";
}

export function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}
