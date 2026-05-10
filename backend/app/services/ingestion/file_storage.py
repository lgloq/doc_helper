from __future__ import annotations

import hashlib
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings

SUPPORTED_FILE_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".csv": "text/csv",
}


@dataclass
class StoredFile:
    original_filename: str
    relative_path: str
    mime_type: str
    file_size: int
    checksum_sha256: str


class LocalDocumentStorage:
    def __init__(self):
        self.settings = get_settings()
        self.base_dir = self.settings.data_dir / "documents"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, document_id, version_number: int, upload_file: UploadFile) -> StoredFile:
        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_FILE_TYPES:
            supported = ", ".join(sorted(SUPPORTED_FILE_TYPES.keys()))
            raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}.")

        safe_name = self._sanitize_filename(upload_file.filename or f"document{suffix}")
        relative_dir = Path(str(document_id)) / f"v{version_number}"
        target_dir = self.base_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name

        upload_file.file.seek(0)
        digest = hashlib.sha256()
        file_size = 0
        with target_path.open("wb") as destination:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
                digest.update(chunk)
                file_size += len(chunk)

        mime_type = upload_file.content_type or SUPPORTED_FILE_TYPES[suffix] or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        return StoredFile(
            original_filename=safe_name,
            relative_path=str(Path("documents") / relative_dir / safe_name),
            mime_type=mime_type,
            file_size=file_size,
            checksum_sha256=digest.hexdigest(),
        )

    def resolve_path(self, relative_path: str) -> Path:
        return self.settings.data_dir / relative_path

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
        return cleaned[:200] or "document"
