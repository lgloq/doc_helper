export type RoleName = "viewer" | "manager" | "admin";
export type DocumentStatus = "draft" | "active" | "archived";
export type IngestStatus = "pending" | "processing" | "ready" | "failed";
export type PrincipalType = "public" | "user" | "role" | "team";
export type MessageRole = "user" | "assistant" | "system";

export interface RoleRead {
  id: string;
  name: RoleName;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface UserRead {
  id: string;
  email: string;
  full_name: string;
  team_name: string | null;
  is_active: boolean;
  role: RoleRead | null;
  created_at: string;
  updated_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: UserRead;
}

export interface DocumentRead {
  id: string;
  title: string;
  description: string | null;
  status: DocumentStatus;
  owner_user_id: string;
  current_version_id: string | null;
  current_user_can_manage: boolean;
  created_at: string;
  updated_at: string;
}

export interface DocumentVersionRead {
  id: string;
  document_id: string;
  version_number: number;
  original_filename: string;
  mime_type: string;
  file_size: number;
  storage_path: string;
  checksum_sha256: string;
  extracted_text: string | null;
  ingest_status: IngestStatus;
  ingest_error: string | null;
  page_count: number | null;
  created_at: string;
  is_current?: boolean;
}

export interface DocumentACLRead {
  id: string;
  document_id: string;
  principal_type: PrincipalType;
  user_id: string | null;
  user_email: string | null;
  role_id: string | null;
  role_name: RoleName | null;
  team_name: string | null;
  can_view: boolean;
  can_manage: boolean;
  created_at: string;
}

export interface DocumentUploadResponse {
  document: DocumentRead;
  version: DocumentVersionRead;
}

export interface IngestionResultRead {
  document_id: string;
  document_version_id: string;
  ingest_status: IngestStatus;
  chunk_count: number;
  page_count: number | null;
}

export interface ChunkRead {
  id: string;
  document_id: string;
  document_version_id: string;
  chunk_index: number;
  content: string;
  preview: string;
  token_count: number;
  section_title: string | null;
  page_number_start: number | null;
  page_number_end: number | null;
  paragraph_start: number | null;
  paragraph_end: number | null;
  char_start: number | null;
  char_end: number | null;
  citation_metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface DocumentDiffChangeRead {
  change_type: string;
  from_paragraph_start: number | null;
  from_paragraph_end: number | null;
  to_paragraph_start: number | null;
  to_paragraph_end: number | null;
  old_text: string | null;
  new_text: string | null;
}

export interface DocumentDiffRead {
  document_id: string;
  from_version_id: string;
  to_version_id: string;
  from_version_number: number;
  to_version_number: number;
  added_count: number;
  deleted_count: number;
  modified_count: number;
  unified_diff: string;
  changes: DocumentDiffChangeRead[];
  impact_hints: string[];
}

export interface DocumentDiffSummaryRead {
  document_id: string;
  from_version_id: string;
  to_version_id: string;
  from_version_number: number;
  to_version_number: number;
  summary: string;
  additions: string[];
  deletions: string[];
  modifications: string[];
  impact_hints: string[];
  summary_provider: string;
  model_name: string | null;
  cache_hit: boolean;
}

export interface SearchDebugInfo {
  accessible_document_count: number;
  lexical_candidate_count: number;
  vector_candidate_count: number;
  fusion_strategy: string;
}

export interface AgentStepRead {
  name: "query_analysis" | "tool_selection" | "tool_execution" | "evidence_review" | "answer_generation";
  input_summary: string;
  output_summary: string;
  status: "completed" | "skipped" | "refused" | string;
  tool_name: string | null;
  metadata: Record<string, unknown>;
}

export interface ChatCitationRead {
  id: string;
  message_id: string;
  chunk_id: string | null;
  document_id: string;
  document_title: string;
  document_version_id: string;
  version_number: number;
  chunk_index: number;
  page_number_start: number | null;
  page_number_end: number | null;
  paragraph_start: number | null;
  paragraph_end: number | null;
  preview: string;
  lexical_score: number | null;
  vector_score: number | null;
  fused_score: number | null;
  rank: number;
  citation_metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface ChatMessageRead {
  id: string;
  session_id: string;
  author_user_id: string | null;
  role: MessageRole;
  content: string;
  model_name: string | null;
  confidence: string | null;
  insufficient_evidence: boolean;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  latency_ms: number | null;
  message_metadata: Record<string, unknown> | null;
  created_at: string;
  citations: ChatCitationRead[];
}

export interface ChatSessionRead {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionDetailRead extends ChatSessionRead {
  messages: ChatMessageRead[];
}

export interface ChatMessageCreateResponse {
  session_id: string;
  user_message: ChatMessageRead;
  assistant_message: ChatMessageRead;
  citations: ChatCitationRead[];
  retrieval_debug: SearchDebugInfo;
}

export interface SourceCitationRead {
  message_citation_id: string | null;
  chunk_id: string | null;
  document_id: string | null;
  document_title: string;
  document_version_id: string | null;
  version_number: number | null;
  chunk_index: number | null;
  page_number_start: number | null;
  page_number_end: number | null;
  paragraph_start: number | null;
  paragraph_end: number | null;
  preview: string;
  fused_score: number | null;
}

export interface TaskItemRead {
  id: string;
  created_by_user_id: string | null;
  source_session_id: string | null;
  source_message_id: string | null;
  title: string;
  description: string | null;
  owner_name: string | null;
  priority: string;
  due_date: string | null;
  status: string;
  source_citations: SourceCitationRead[] | null;
  created_at: string;
  updated_at: string;
}

export interface TaskExtractResponse {
  items: TaskItemRead[];
}

export interface WeeklyReportDraftRead {
  id: string;
  created_by_user_id: string | null;
  source_session_id: string | null;
  title: string;
  summary: string | null;
  completed_this_week: string[];
  risks_blockers: string[];
  next_week_plan: string[];
  reference_sources: SourceCitationRead[];
  source_message_ids: string[];
  status: string;
  created_at: string;
  updated_at: string;
}

export interface WeeklyReportGenerateResponse {
  report: WeeklyReportDraftRead;
}

export interface FAQEntryRead {
  id: string;
  created_by_user_id: string | null;
  source_session_id: string | null;
  source_message_id: string | null;
  question: string;
  answer: string;
  quality: string;
  status: string;
  source_citations: SourceCitationRead[];
  created_at: string;
  updated_at: string;
}

export interface FAQGenerateResponse {
  entries: FAQEntryRead[];
}

export interface EvalRunRead {
  id: string;
  dataset_name: string;
  status: string;
  total_cases: number;
  started_at: string | null;
  finished_at: string | null;
  summary_json: Record<string, unknown> | null;
  error_text: string | null;
  created_at: string;
  updated_at: string;
}

export interface EvalResultRowRead {
  id: string;
  run_id: string;
  case_id: string;
  acting_user_email: string;
  retrieval_hit_rate: number;
  citation_accuracy: number;
  answer_faithfulness: number;
  permission_isolation_correct: boolean;
  overall_pass: boolean;
  details_json: Record<string, unknown>;
  created_at: string;
}

export interface EvalRunDetailRead extends EvalRunRead {
  results: EvalResultRowRead[];
}

export interface TraceLogRead {
  id: string;
  trace_type: string;
  user_id: string | null;
  session_id: string | null;
  user_message_id: string | null;
  assistant_message_id: string | null;
  query_text: string | null;
  retrieved_chunks_json: Record<string, unknown>[];
  selected_citations_json: Record<string, unknown>[];
  model_name: string | null;
  latency_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  error_text: string | null;
  trace_metadata: Record<string, unknown> | null;
  created_at: string;
}
