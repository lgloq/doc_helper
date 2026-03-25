from __future__ import annotations

from enum import Enum


class RoleName(str, Enum):
    VIEWER = "viewer"
    MANAGER = "manager"
    ADMIN = "admin"


class DocumentStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class IngestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class PrincipalType(str, Enum):
    PUBLIC = "public"
    USER = "user"
    ROLE = "role"
    TEAM = "team"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
