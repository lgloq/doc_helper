from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.search import SearchDebugInfo
from app.schemas.workflow import SourceCitationRead

CopilotIntent = Literal[
    "document_qa",
    "topic_qa",
    "version_compare",
    "workflow_generation",
    "unsupported_or_unclear",
]
AgentStepName = Literal[
    "query_analysis",
    "tool_selection",
    "tool_execution",
    "evidence_review",
    "answer_generation",
]
AgentStepStatus = Literal["completed", "skipped", "refused"]

ArtifactType = Literal["tasks", "weekly_report", "faq"]
QAAnswerType = Literal["grounded_answer", "refusal"]
VersionCompareAnswerType = Literal["version_compare_result", "refusal"]
WorkflowAnswerType = Literal["workflow_result", "refusal"]
ConfidenceLabel = Literal["high", "medium", "low", "insufficient"]


class RouterAccessibleDocument(BaseModel):
    document_id: UUID
    title: str


class RouterDecision(BaseModel):
    intent: CopilotIntent
    target_document_id: UUID | None = None
    target_document_title: str | None = None
    requested_document_name: str | None = None
    topic: str | None = None
    artifact_type: ArtifactType | None = None
    from_version_ref: str | None = None
    to_version_ref: str | None = None
    needs_citations: bool = True
    should_refuse_if_inaccessible: bool = False
    reasoning_brief: str = Field(default="")


class RouterDecisionResult(BaseModel):
    decision: RouterDecision
    provider_name: str
    model_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    raw_payload: dict[str, Any] | None = None


class AgentStep(BaseModel):
    name: AgentStepName
    input_summary: str
    output_summary: str
    status: AgentStepStatus
    tool_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCitation(BaseModel):
    chunk_id: UUID | None = None
    document_id: UUID | None = None
    document_title: str
    document_version_id: UUID | None = None
    version_number: int | None = None
    chunk_index: int | None = None
    page_number_start: int | None = None
    page_number_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    preview: str
    fused_score: float | None = None


class QAAnswerResult(BaseModel):
    answer_type: QAAnswerType
    answer: str
    confidence: ConfidenceLabel
    citations: list[ToolCitation] = Field(default_factory=list)
    refusal_reason: str | None = None
    target_document: str | None = None
    intent: CopilotIntent


class VersionCompareResult(BaseModel):
    answer_type: VersionCompareAnswerType
    answer: str
    confidence: ConfidenceLabel
    intent: Literal["version_compare"] = "version_compare"
    target_document: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    summary: str | None = None
    additions: list[str] = Field(default_factory=list)
    deletions: list[str] = Field(default_factory=list)
    modifications: list[str] = Field(default_factory=list)
    impact_hints: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None


class WorkflowGenerationResult(BaseModel):
    answer_type: WorkflowAnswerType
    answer: str
    confidence: ConfidenceLabel
    intent: Literal["workflow_generation"] = "workflow_generation"
    artifact_type: ArtifactType | None = None
    structured_payload: dict[str, Any] | None = None
    citations: list[SourceCitationRead] = Field(default_factory=list)
    refusal_reason: str | None = None


class CopilotExecutionMetadata(BaseModel):
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output_summary: dict[str, Any] = Field(default_factory=dict)
    retrieval_debug: SearchDebugInfo | None = None

