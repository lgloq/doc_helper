from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from app.core.config import get_settings  # noqa: E402


def wait_for_database(max_attempts: int = 60, sleep_seconds: int = 2) -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            print("Database is ready.")
            return
        except Exception as exc:  # pragma: no cover - runtime convenience script
            print(f"Waiting for database ({attempt}/{max_attempts}): {exc}")
            time.sleep(sleep_seconds)
    raise SystemExit("Database did not become ready in time.")


def main() -> None:
    wait_for_database()
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True, cwd=BACKEND_DIR)

    settings = get_settings()
    command = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    if settings.debug:
        command.append("--reload")
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
