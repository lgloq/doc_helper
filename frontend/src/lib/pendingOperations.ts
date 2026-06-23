export interface PendingChatOperation {
  id: string;
  type: "chat_message";
  sessionId: string;
  content: string;
  topK: number;
  createdAt: number;
  updatedAt: number;
  lastError?: string;
}

export interface PendingEvalOperation {
  id: string;
  type: "eval_run";
  datasetName: string;
  topK: number;
  seedDemoCases: boolean;
  createdAt: number;
  updatedAt: number;
  lastError?: string;
}

type PendingOperation = PendingChatOperation | PendingEvalOperation;

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
  return isPendingChatOperation(value) || isPendingEvalOperation(value);
}

function isPendingChatOperation(value: unknown): value is PendingChatOperation {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Partial<PendingChatOperation>;
  return (
    item.type === "chat_message" &&
    typeof item.id === "string" &&
    typeof item.sessionId === "string" &&
    typeof item.content === "string" &&
    typeof item.topK === "number"
  );
}

function isPendingEvalOperation(value: unknown): value is PendingEvalOperation {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Partial<PendingEvalOperation>;
  return (
    item.type === "eval_run" &&
    typeof item.id === "string" &&
    typeof item.datasetName === "string" &&
    typeof item.topK === "number" &&
    typeof item.seedDemoCases === "boolean"
  );
}

function createOperationId(prefix: "chat" | "eval"): string {
  if (typeof window.crypto?.randomUUID === "function") {
    return `${prefix}-${window.crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
  writePendingOperations([...readPendingOperations().filter((item) => item.id !== operation.id), operation]);
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
  writePendingOperations([
    ...readPendingOperations().filter(
      (item) => item.id !== operation.id && !(item.type === "eval_run" && item.datasetName === operation.datasetName),
    ),
    operation,
  ]);
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

function touchPendingOperation(operationId: string, lastError?: string) {
  const items = readPendingOperations();
  writePendingOperations(
    items.map((item) =>
      item.id === operationId ? { ...item, updatedAt: Date.now(), lastError: lastError ?? item.lastError } : item,
    ),
  );
}

function removePendingOperation(operationId: string) {
  writePendingOperations(readPendingOperations().filter((item) => item.id !== operationId));
}