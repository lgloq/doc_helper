export interface PendingOperationBase {
  id: string;
  createdAt: number;
  updatedAt: number;
  jobId?: string;
  lastError?: string;
}

export interface PendingChatOperation extends PendingOperationBase {
  type: "chat_message";
  sessionId: string;
  content: string;
  topK: number;
}

export interface PendingEvalOperation extends PendingOperationBase {
  type: "eval_run";
  datasetName: string;
  topK: number;
  seedDemoCases: boolean;
}

export interface PendingDiffSummaryOperation extends PendingOperationBase {
  type: "document_diff_summary";
  documentId: string;
  fromVersionId: string;
  toVersionId: string;
  forceRefresh: boolean;
}

export interface PendingDocumentIngestOperation extends PendingOperationBase {
  type: "document_ingest";
  documentId: string;
  versionId: string;
}

type PendingOperation =
  | PendingChatOperation
  | PendingEvalOperation
  | PendingDiffSummaryOperation
  | PendingDocumentIngestOperation;

const STORAGE_KEY = "eka_pending_operations_v1";

function readPendingOperations(): PendingOperation[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isPendingOperation);
  } catch {
    return [];
  }
}

function writePendingOperations(items: PendingOperation[]) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

function isPendingOperation(value: unknown): value is PendingOperation {
  return (
    isPendingChatOperation(value) ||
    isPendingEvalOperation(value) ||
    isPendingDiffSummaryOperation(value) ||
    isPendingDocumentIngestOperation(value)
  );
}

function isPendingBaseOperation(value: unknown): value is PendingOperationBase {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Partial<PendingOperationBase>;
  return typeof item.id === "string" && typeof item.createdAt === "number" && typeof item.updatedAt === "number";
}

function isPendingChatOperation(value: unknown): value is PendingChatOperation {
  if (!isPendingBaseOperation(value) || (value as Partial<PendingChatOperation>).type !== "chat_message") {
    return false;
  }
  const item = value as Partial<PendingChatOperation>;
  return typeof item.sessionId === "string" && typeof item.content === "string" && typeof item.topK === "number";
}

function isPendingEvalOperation(value: unknown): value is PendingEvalOperation {
  if (!isPendingBaseOperation(value) || (value as Partial<PendingEvalOperation>).type !== "eval_run") {
    return false;
  }
  const item = value as Partial<PendingEvalOperation>;
  return typeof item.datasetName === "string" && typeof item.topK === "number" && typeof item.seedDemoCases === "boolean";
}

function isPendingDiffSummaryOperation(value: unknown): value is PendingDiffSummaryOperation {
  if (!isPendingBaseOperation(value) || (value as Partial<PendingDiffSummaryOperation>).type !== "document_diff_summary") {
    return false;
  }
  const item = value as Partial<PendingDiffSummaryOperation>;
  return (
    typeof item.documentId === "string" &&
    typeof item.fromVersionId === "string" &&
    typeof item.toVersionId === "string" &&
    typeof item.forceRefresh === "boolean"
  );
}

function isPendingDocumentIngestOperation(value: unknown): value is PendingDocumentIngestOperation {
  if (!isPendingBaseOperation(value) || (value as Partial<PendingDocumentIngestOperation>).type !== "document_ingest") {
    return false;
  }
  const item = value as Partial<PendingDocumentIngestOperation>;
  return typeof item.documentId === "string" && typeof item.versionId === "string";
}

function createOperationId(prefix: "chat" | "eval" | "diff" | "ingest"): string {
  if (typeof window.crypto?.randomUUID === "function") {
    return `${prefix}-${window.crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function upsertPendingOperation(operation: PendingOperation) {
  writePendingOperations([...readPendingOperations().filter((item) => item.id !== operation.id), operation]);
}

function updatePendingOperation(operationId: string, updater: (item: PendingOperation) => PendingOperation) {
  const items = readPendingOperations();
  writePendingOperations(items.map((item) => (item.id === operationId ? updater(item) : item)));
}

export function createPendingChatOperation(input: { sessionId: string; content: string; topK: number }): PendingChatOperation {
  const now = Date.now();
  const operation: PendingChatOperation = {
    id: createOperationId("chat"),
    type: "chat_message",
    sessionId: input.sessionId,
    content: input.content,
    topK: input.topK,
    createdAt: now,
    updatedAt: now,
  };
  upsertPendingOperation(operation);
  return operation;
}

export function listPendingChatOperations(): PendingChatOperation[] {
  return readPendingOperations().filter((item): item is PendingChatOperation => item.type === "chat_message");
}

export function touchPendingChatOperation(operationId: string, lastError?: string) {
  touchPendingOperation(operationId, lastError);
}

export function removePendingChatOperation(operationId: string) {
  removePendingOperation(operationId);
}

export function createPendingEvalOperation(input: {
  datasetName: string;
  topK: number;
  seedDemoCases: boolean;
}): PendingEvalOperation {
  const now = Date.now();
  const operation: PendingEvalOperation = {
    id: createOperationId("eval"),
    type: "eval_run",
    datasetName: input.datasetName,
    topK: input.topK,
    seedDemoCases: input.seedDemoCases,
    createdAt: now,
    updatedAt: now,
  };
  upsertPendingOperation(operation);
  return operation;
}

export function listPendingEvalOperations(datasetName?: string): PendingEvalOperation[] {
  return readPendingOperations().filter(
    (item): item is PendingEvalOperation => item.type === "eval_run" && (!datasetName || item.datasetName === datasetName),
  );
}

export function touchPendingEvalOperation(operationId: string, lastError?: string) {
  touchPendingOperation(operationId, lastError);
}

export function removePendingEvalOperation(operationId: string) {
  removePendingOperation(operationId);
}

export function createPendingDiffSummaryOperation(input: {
  documentId: string;
  fromVersionId: string;
  toVersionId: string;
  forceRefresh: boolean;
}): PendingDiffSummaryOperation {
  const now = Date.now();
  const operation: PendingDiffSummaryOperation = {
    id: createOperationId("diff"),
    type: "document_diff_summary",
    documentId: input.documentId,
    fromVersionId: input.fromVersionId,
    toVersionId: input.toVersionId,
    forceRefresh: input.forceRefresh,
    createdAt: now,
    updatedAt: now,
  };
  upsertPendingOperation(operation);
  return operation;
}

export function listPendingDiffSummaryOperations(documentId?: string): PendingDiffSummaryOperation[] {
  return readPendingOperations().filter(
    (item): item is PendingDiffSummaryOperation =>
      item.type === "document_diff_summary" && (!documentId || item.documentId === documentId),
  );
}

export function touchPendingDiffSummaryOperation(operationId: string, lastError?: string) {
  touchPendingOperation(operationId, lastError);
}

export function removePendingDiffSummaryOperation(operationId: string) {
  removePendingOperation(operationId);
}

export function createPendingDocumentIngestOperation(input: {
  documentId: string;
  versionId: string;
}): PendingDocumentIngestOperation {
  const now = Date.now();
  const operation: PendingDocumentIngestOperation = {
    id: createOperationId("ingest"),
    type: "document_ingest",
    documentId: input.documentId,
    versionId: input.versionId,
    createdAt: now,
    updatedAt: now,
  };
  upsertPendingOperation(operation);
  return operation;
}

export function listPendingDocumentIngestOperations(documentId?: string): PendingDocumentIngestOperation[] {
  return readPendingOperations().filter(
    (item): item is PendingDocumentIngestOperation =>
      item.type === "document_ingest" && (!documentId || item.documentId === documentId),
  );
}

export function touchPendingDocumentIngestOperation(operationId: string, lastError?: string) {
  touchPendingOperation(operationId, lastError);
}

export function removePendingDocumentIngestOperation(operationId: string) {
  removePendingOperation(operationId);
}

export function setPendingOperationJob(operationId: string, jobId: string) {
  updatePendingOperation(operationId, (item) => ({ ...item, jobId, updatedAt: Date.now() }));
}

function touchPendingOperation(operationId: string, lastError?: string) {
  updatePendingOperation(operationId, (item) => ({
    ...item,
    updatedAt: Date.now(),
    lastError: lastError ?? item.lastError,
  }));
}

function removePendingOperation(operationId: string) {
  writePendingOperations(readPendingOperations().filter((item) => item.id !== operationId));
}
