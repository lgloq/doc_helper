import type { ChatCitationRead, SourceCitationRead } from "../types/api";
import { locationLabel } from "../lib/format";
import { StatusBadge } from "./StatusBadge";

type Citation = ChatCitationRead | SourceCitationRead;

interface CitationListProps {
  citations: Citation[];
  selectedCitationId?: string | null;
  onSelect?: (citation: Citation) => void;
  title?: string;
}

export function CitationList({ citations, selectedCitationId, onSelect, title = "Citations" }: CitationListProps) {
  if (!citations.length) {
    return (
      <section className="panel">
        <div className="panel-header">
          <h3>{title}</h3>
        </div>
        <p className="muted">No citations attached.</p>
      </section>
    );
  }

  return (
    <section className="panel citation-panel">
      <div className="panel-header">
        <h3>{title}</h3>
        <StatusBadge tone="info">{citations.length} sources</StatusBadge>
      </div>
      <div className="citation-list">
        {citations.map((citation, index) => {
          const citationId = "id" in citation ? citation.id : citation.message_citation_id ?? `${citation.document_title}-${index}`;
          const isSelected = selectedCitationId === citationId;
          return (
            <button
              key={citationId}
              className={`citation-card ${isSelected ? "is-selected" : ""}`}
              onClick={() => onSelect?.(citation)}
              type="button"
            >
              <div className="citation-card-topline">
                <strong>{citation.document_title}</strong>
                <span>
                  v{citation.version_number ?? "?"} · {locationLabel(citation)}
                </span>
              </div>
              <p>{citation.preview}</p>
            </button>
          );
        })}
      </div>
    </section>
  );
}
